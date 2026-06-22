#!/usr/bin/env bash
#
# exopipe container entrypoint -- one image, two Cloud Run shapes.
#
#   * dashboard (DEFAULT) -> Streamlit service bound to $PORT (Cloud Run service)
#   * job | run           -> `exopipe run ...` sector-scale batch (Cloud Run Job)
#   * demo                -> `exopipe demo ...` offline synthetic end-to-end run
#   * <anything else>     -> forwarded straight to the `exopipe` CLI
#
# The dispatch key is the first positional arg, or $EXOPIPE_MODE when no args are
# given (handy for a Cloud Run *service* whose container has no `args`).
set -euo pipefail

mode="${1:-${EXOPIPE_MODE:-dashboard}}"

# Consume the dispatch token so the remainder ("$@") is the command's own args.
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "$mode" in
  dashboard)
    # A freshly deployed service shows the bundled example results immediately.
    exec streamlit run app/dashboard.py \
      --server.port "${PORT:-8080}" \
      --server.address 0.0.0.0 \
      --server.headless true \
      -- \
      --catalog "${EXOPIPE_CATALOG:-examples/example_catalog.csv}" \
      --figdir "${EXOPIPE_FIGDIR:-examples}"
    ;;
  job | run)
    # Cloud Run Job batch mode: `exopipe run --input ... [flags]`.
    exec exopipe run "$@"
    ;;
  demo)
    exec exopipe demo "$@"
    ;;
  *)
    # Pass through to the CLI (e.g. `version`, `report`, `fetch`, `train`).
    exec exopipe "$mode" "$@"
    ;;
esac
