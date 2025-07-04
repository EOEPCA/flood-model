cwlVersion: v1.2
s:softwareVersion: "0.3.0"

$namespaces:
  s: https://schema.org/
schemas:
  - http://schema.org/version/9.0/schemaorg-current-http.rdf

$graph:
  - class: Workflow
    id: flood-model-inference
    label: Flood detection model inference
    doc: Run flood detection model inference on STAC catalog
    inputs:
      stac_item:
        type: string
        label: STAC item URL
        doc: URL of a STAC item
    outputs:
      stac_catalog:
        type: Directory
        outputSource: node_inference/stac_catalog

    steps:
      node_download_stac:
        run: "#download-stac"
        in:
          stac_item: stac_item
        out:
          - assets_dir
      node_inference:
        run: "#inference"
        in:
          input_dir: node_download_stac/assets_dir
        out:
          - stac_catalog

  - class: CommandLineTool
    id: inference
    label: Flood detection model inference
    doc: Run flood detection model inference image
    baseCommand: ["python3", "/app/inference.py"]
    arguments: ["/app/model.onnx", "--stac-output"]
    hints:
      DockerRequirement:
        dockerPull: eoepca/flood-detection-model:v2-0.3.0
    requirements:
      ResourceRequirement:
        coresMin: 1
        coresMax: 2
        ramMin: 2048
        ramMax: 4096
    inputs:
      input_dir:
        type: Directory
        inputBinding:
          position: 1
    outputs:
      stac_catalog:
        type: Directory
        outputBinding:
          glob: "predictions"

  - class: CommandLineTool
    id: download-stac
    label: Download STAC assets
    doc: Receive STAC item URL and dump the assets in output directory
    baseCommand: ["uvx", "eodag", "download"]
    arguments: ["--output-dir", "_assets"]
    hints:
      DockerRequirement:
        dockerPull: astral/uv:0.7-python3.12-bookworm-slim
    requirements:
      ResourceRequirement:
        coresMin: 1
        coresMax: 1
        ramMin: 512
        ramMax: 512
      NetworkAccess:
        networkAccess: true
    inputs:
      stac_item:
        type: string
        inputBinding:
          prefix: --stac-item
    outputs:
      assets_dir:
        type: Directory
        outputBinding:
          glob: "_assets"
