"""
Stage 6: Config check — cross-reference the host/literal extracted for each
external-call candidate against config-like files in the repo (yaml/json/
.env/Dockerfiles/k8s manifests etc). If we find the host, or the first
label of the host (a plausible k8s service name), mentioned in a config
file, we link the external service to that config file.
"""
import re

from .obs import get_logger, log, TRACE_EDGES

LOG = get_logger("config")

_LABEL_RE = re.compile(r'^[A-Za-z0-9][\w.-]*$')


def check_config_references(edges, config_records, ctx=None, pass_label="initial"):
    """Returns list of (edge, config_path) matches; does not mutate edges
    beyond attaching config_matches for convenience.

    Safe to call more than once on the same edge list — e.g. once after the
    literal/const-prop checks, and again after the Joern escalation and LLM
    fallback stages resolve additional hosts — since already-recorded
    (edge, config_path) matches are skipped rather than duplicated.
    """
    if not config_records:
        # Worth a warning rather than a silent no-op: an empty result here
        # means "we never looked", not "we looked and found nothing".
        log(LOG, "warning", "no config files in repo; config stage is a no-op",
            pass_label=pass_label)
        if ctx is not None and pass_label == "initial":
            ctx.note("warning", "config",
                     "No config/manifest files were detected, so external services "
                     "could not be cross-referenced to any deployment config. "
                     "Check CONFIG_EXTS/CONFIG_NAME_HINTS if the repo does have them.")
        return []

    matches = []
    hosts_checked = 0
    for edge in edges:
        if edge.status != "external_candidate" or not edge.host:
            continue
        hosts_checked += 1
        host = edge.host
        service_label = host.split(".")[0] if _LABEL_RE.match(host.split(":")[0]) else None
        needles = {host}
        if service_label and len(service_label) > 2:
            needles.add(service_label)

        for cfg in config_records:
            if cfg.path in edge.config_matches:
                continue
            text = "\n".join(cfg.source_lines)
            hit = next((n for n in needles if n and n.lower() in text.lower()), None)
            if hit:
                edge.config_matches.append(cfg.path)
                matches.append((edge, cfg.path, hit))
                if TRACE_EDGES:
                    log(LOG, "debug", "host matched a config file",
                        host=host, config=cfg.path, needle=hit)

    if ctx is not None:
        ctx.bump(f"config.pass.{pass_label}.hosts_checked", hosts_checked)
        ctx.bump(f"config.pass.{pass_label}.matches", len(matches))
    log(LOG, "info", "config cross-reference complete", pass_label=pass_label,
        config_files=len(config_records), hosts_checked=hosts_checked,
        new_matches=len(matches))
    return matches
