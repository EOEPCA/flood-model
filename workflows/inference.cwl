cwlVersion: v1.2
s:softwareVersion: "0.4.0"
s:license: https://spdx.org/licenses/Apache-2.0
s:codeRepository: https://github.com/EOEPCA/flood-model

$namespaces:
  s: https://schema.org/
schemas:
  - http://schema.org/version/9.0/schemaorg-current-http.rdf

$graph:
  - class: Workflow
    id: flood-detection-model-inference
    label: Flood detection model inference
    doc: Run flood detection model inference on STAC catalog
    inputs:
      stac_item:
        type: Directory
        label: STAC item reference
        doc: Reference to a STAC item
    outputs:
      stac_catalog:
        type: Directory
        outputSource: node_stac_cataloging/stac_catalog

    steps:
      node_stac_prepare:
        run: "#stac-prepare"
        in:
          stac_item: stac_item
        out:
          - assets_dir
      node_inference:
        run: "#inference"
        in:
          input_dir: node_stac_prepare/assets_dir
        out:
          - output_dir
      node_stac_cataloging:
        run: "#stac-cataloging"
        in:
          input_dir: node_inference/output_dir
        out:
          - stac_catalog

  - class: CommandLineTool
    id: inference
    label: Flood detection model inference
    doc: Run flood detection model inference image
    baseCommand: ["python3", "/app/inference.py"]
    arguments: ["/app/model.onnx"]
    hints:
      DockerRequirement:
        dockerPull: eoepca/flood-detection-model:v2-0.4.0
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
      output_dir:
        type: Directory
        outputBinding:
          glob: "predictions"

  - class: CommandLineTool
    id: stac-prepare
    label: STAC preparation
    doc: Receive STAC item/catalog and dump the assets in directory
    baseCommand: ["python", "-m", "eoap"]
    arguments: ["--prepare", "--output", "_assets"]
    hints:
      DockerRequirement:
        dockerPull: eoepca/cwl-eoap:0.1.0
    requirements:
      EnvVarRequirement:
        envDef:
          PATH: /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
          PYTHONPATH: /app
      ResourceRequirement:
        coresMin: 1
        coresMax: 1
        ramMin: 512
        ramMax: 512
    inputs:
      stac_item:
        type: Directory
        inputBinding:
          position: 1
    outputs:
      assets_dir:
        type: Directory
        outputBinding:
          glob: "_assets"

  - class: CommandLineTool
    id: stac-cataloging
    label: STAC cataloging
    doc: Create STAC catalog from directory of files
    baseCommand: ["python", "-m", "eoap"]
    arguments: ["--catalog", "--output", "_stac_catalog"]
    hints:
      DockerRequirement:
        dockerPull: eoepca/cwl-eoap:0.1.0
    requirements:
      EnvVarRequirement:
        envDef:
          PATH: /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
          PYTHONPATH: /app
      ResourceRequirement:
        coresMin: 1
        coresMax: 1
        ramMin: 512
        ramMax: 512
    inputs:
      input_dir:
        type: Directory
        inputBinding:
          position: 1
    outputs:
      stac_catalog:
        type: Directory
        outputBinding:
          glob: "_stac_catalog"
