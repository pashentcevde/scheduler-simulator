"""Быстрый путь для `kubectl get ... -o json` через `kubectl proxy`.

Зачем: снапшотер на каждом такте делает два вызова kubectl, каждый из которых
это форк процесса, чтение kubeconfig и TLS-хендшейк — порядка 100-300 мс.
При такте 1 с это терпимо, при 0.25 с (нужном для больших time_scale) — нет.

Здесь тот же самый запрос уходит обычным HTTP на локальный `kubectl proxy`,
который держит одно соединение с apiserver'ом. Форк и хендшейк исчезают,
остаётся сериализация ответа.

Использование: ничего не меняется, `kubectl_json()` из lab.py сам берёт этот
путь, если прокси доступен, и молча откатывается на subprocess, если нет.

Переменные окружения:
    KUBE_PROXY_URL        адрес уже поднятого прокси (например
                          http://127.0.0.1:8001). Если задан, свой прокси
                          не поднимается и не гасится.
    KUBE_PROXY_AUTOSTART  0 — не поднимать прокси самому (по умолчанию 1).
    KUBE_LIST_FROM_CACHE  1 — добавлять resourceVersion=0 к спискам. Ответ
                          отдаётся из watch-кэша apiserver'а, а не из etcd:
                          заметно быстрее, но данные могут отставать на
                          доли секунды. Для снапшотера это приемлемо,
                          для решений планировщика — нет.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

# ── карта ресурсов ────────────────────────────────────────────────────────────
# (группа/версия, множественное имя, kind, namespaced)
# Только то, что стенд реально запрашивает. Незнакомый ресурс -> откат на
# subprocess, поэтому список можно не расширять «на всякий случай».
RESOURCES: dict[str, tuple[str, str, str, bool]] = {
    "pods": ("api/v1", "pods", "Pod", True),
    "po": ("api/v1", "pods", "Pod", True),
    "nodes": ("api/v1", "nodes", "Node", False),
    "no": ("api/v1", "nodes", "Node", False),
    "events": ("api/v1", "events", "Event", True),
    "jobs": ("apis/batch/v1", "jobs", "Job", True),
    "clusterqueues": (None, "clusterqueues", "ClusterQueue", False),
    "workloads": (None, "workloads", "Workload", True),
    "localqueues": (None, "localqueues", "LocalQueue", True),
    "resourceflavors": (None, "resourceflavors", "ResourceFlavor", False),
}

KUEUE_GROUP = "kueue.x-k8s.io"
KUEUE_VERSION_FALLBACKS = ("v1beta2", "v1beta1")


class ProxyUnavailable(RuntimeError):
    """Прокси не поднялся или ответил не так — вызывающий откатывается."""


class KubeProxy:
    """Один процесс `kubectl proxy` на весь прогон плюс HTTP-сессия к нему."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._base: str | None = None
        self._proc: subprocess.Popen | None = None
        self._session = None
        self._kueue_gv: str | None = None
        self._disabled = False
        self._consecutive = 0
        self._stats = {"calls": 0, "seconds": 0.0, "fallbacks": 0}

    # ── жизненный цикл ────────────────────────────────────────────────────────
    def base_url(self) -> str:
        if self._disabled:
            raise ProxyUnavailable("быстрый путь отключён")
        if self._base:
            return self._base
        with self._lock:
            if self._base:
                return self._base
            env = os.environ.get("KUBE_PROXY_URL", "").strip().rstrip("/")
            if env:
                self._base = env
            elif os.environ.get("KUBE_PROXY_AUTOSTART", "1") != "0":
                self._base = self._spawn()
            else:
                self._disabled = True
                raise ProxyUnavailable("KUBE_PROXY_URL не задан, автостарт выключен")
            self._open_session()
            return self._base

    def _spawn(self) -> str:
        """Поднять свой прокси на свободном порту и дождаться готовности."""
        ctx = os.environ.get("KUBE_CONTEXT", "")
        cmd = ["kubectl", *(["--context", ctx] if ctx else []), "proxy", "--port=0"]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except OSError as exc:
            self._disabled = True
            raise ProxyUnavailable(f"не смог запустить kubectl proxy: {exc}") from exc

        # kubectl печатает "Starting to serve on 127.0.0.1:PORT"
        deadline = time.time() + 15.0
        addr = None
        while time.time() < deadline:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    break
                continue
            m = re.search(r"Starting to serve on\s+(\S+)", line)
            if m:
                addr = m.group(1)
                break
        if not addr:
            proc.kill()
            self._disabled = True
            raise ProxyUnavailable("kubectl proxy не сообщил адрес за 15 секунд")

        self._proc = proc
        atexit.register(self.close)
        print(f"[kube-proxy] поднят на {addr}", file=sys.stderr)
        return f"http://{addr}"

    def _open_session(self) -> None:
        try:
            import requests  # noqa: PLC0415 — опциональная зависимость
            from requests.adapters import HTTPAdapter  # noqa: PLC0415
        except ImportError:
            self._session = None  # откатываемся на urllib, keep-alive не будет
            return
        sess = requests.Session()
        # пул побольше: снапшотер и модель GPU могут ходить одновременно
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        sess.mount("http://", adapter)
        self._session = sess

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass
            self._session = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.kill()
            self._proc = None
            s = self._stats
            if s["calls"]:
                print(
                    f"[kube-proxy] {s['calls']} запросов, "
                    f"в среднем {1000 * s['seconds'] / s['calls']:.0f} мс, "
                    f"откатов на kubectl: {s['fallbacks']}",
                    file=sys.stderr,
                )

    # ── запросы ───────────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url()}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        started = time.time()
        if self._session is not None:
            resp = self._session.get(url, timeout=30)
            if resp.status_code != 200:
                raise ProxyUnavailable(f"{url} -> HTTP {resp.status_code}")
            data = resp.json()
        else:
            with urllib.request.urlopen(url, timeout=30) as fh:
                data = json.loads(fh.read().decode("utf-8"))
        self._stats["calls"] += 1
        self._stats["seconds"] += time.time() - started
        return data

    def _kueue_group_version(self) -> str:
        """Версия API kueue: спрашиваем один раз, дальше из памяти."""
        if self._kueue_gv:
            return self._kueue_gv
        try:
            data = self._get(f"apis/{KUEUE_GROUP}", {})
            gv = (data.get("preferredVersion") or {}).get("groupVersion")
            if gv:
                self._kueue_gv = f"apis/{gv}"
                return self._kueue_gv
        except Exception:  # noqa: BLE001 — падать из-за discovery не стоит
            pass
        self._kueue_gv = f"apis/{KUEUE_GROUP}/{KUEUE_VERSION_FALLBACKS[0]}"
        return self._kueue_gv

    def list(
        self,
        resource: str,
        all_namespaces: bool,
        namespace: str | None,
        label_selector: str | None,
        field_selector: str | None = None,
    ) -> list[dict]:
        prefix, plural, kind, namespaced = RESOURCES[resource]
        if prefix is None:
            prefix = self._kueue_group_version()

        if namespaced and namespace and not all_namespaces:
            path = f"{prefix}/namespaces/{namespace}/{plural}"
        else:
            path = f"{prefix}/{plural}"

        params: dict[str, str] = {}
        if label_selector:
            params["labelSelector"] = label_selector
        if field_selector:
            params["fieldSelector"] = field_selector
        if os.environ.get("KUBE_LIST_FROM_CACHE", "") == "1":
            params["resourceVersion"] = "0"

        data = self._get(path, params)
        items = data.get("items") or []
        # kubectl проставляет kind/apiVersion каждому элементу при выводе
        # нескольких типов; потребители в стенде на это опираются
        api_version = prefix.replace("apis/", "", 1) if prefix.startswith("apis/") else "v1"
        for it in items:
            it.setdefault("kind", kind)
            it.setdefault("apiVersion", api_version)
        return items

    def note_fallback(self) -> None:
        """Считаем откаты подряд: если прокси умер, не стоит на каждом такте
        ходить в него впустую — тогда быстрый путь стоит дороже обычного."""
        self._stats["fallbacks"] += 1
        self._consecutive += 1
        if self._consecutive >= 5 and not self._disabled:
            self._disabled = True
            print(
                "[kube-proxy] пять неудач подряд, выключаю быстрый путь "
                "до конца прогона (работаем через kubectl)",
                file=sys.stderr,
            )

    def note_success(self) -> None:
        self._consecutive = 0


PROXY = KubeProxy()


# ── разбор аргументов kubectl ─────────────────────────────────────────────────
def parse_get_args(args: tuple[str, ...]) -> dict | None:
    """Разложить `("get", "pods,jobs", "-A", "-l", "sel")` на параметры запроса.

    None — форму не распознали (есть флаг, который мы не воспроизводим),
    вызывающий должен откатиться на subprocess.
    """
    if not args or args[0] != "get" or len(args) < 2:
        return None
    resources = [r.strip() for r in args[1].split(",") if r.strip()]
    if not resources or any(r not in RESOURCES for r in resources):
        return None

    out = {
        "resources": resources,
        "all_namespaces": False,
        "namespace": None,
        "label_selector": None,
        "field_selector": None,
    }
    i = 2
    while i < len(args):
        a = args[i]
        if a in ("-A", "--all-namespaces"):
            out["all_namespaces"] = True
            i += 1
        elif a in ("-l", "--selector"):
            if i + 1 >= len(args):
                return None
            out["label_selector"] = args[i + 1]
            i += 2
        elif a.startswith("--selector="):
            out["label_selector"] = a.split("=", 1)[1]
            i += 1
        elif a in ("-n", "--namespace"):
            if i + 1 >= len(args):
                return None
            out["namespace"] = args[i + 1]
            i += 2
        elif a == "--field-selector":
            if i + 1 >= len(args):
                return None
            out["field_selector"] = args[i + 1]
            i += 2
        elif a in ("-o", "--output"):
            # `-o json` добавляет kubectl_json, всё остальное не наш случай
            if i + 1 >= len(args) or args[i + 1] != "json":
                return None
            i += 2
        else:
            return None  # незнакомый флаг — не рискуем
    return out


def try_get_json(args: tuple[str, ...]) -> dict | None:
    """Выполнить `get` через прокси. None — не смогли, нужен откат."""
    parsed = parse_get_args(args)
    if parsed is None:
        return None
    try:
        items: list[dict] = []
        for res in parsed["resources"]:
            items.extend(
                PROXY.list(
                    res,
                    parsed["all_namespaces"],
                    parsed["namespace"],
                    parsed["label_selector"],
                    parsed["field_selector"],
                )
            )
    except ProxyUnavailable:
        PROXY.note_fallback()
        return None
    except Exception as exc:  # noqa: BLE001 — любая сетевая беда = откат
        PROXY.note_fallback()
        print(f"[kube-proxy] запрос не удался ({exc}), откат на kubectl", file=sys.stderr)
        return None
    PROXY.note_success()
    return {"apiVersion": "v1", "kind": "List", "items": items}
