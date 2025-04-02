cwlVersion: v1.2
s:softwareVersion: "0.1.0"

$namespaces:
  s: https://schema.org/
schemas:
- http://schema.org/version/9.0/schemaorg-current-http.rdf

$graph:
- class: Workflow
  id: main
  label: Flood detection model inference
  doc: Run flood detection model inference image
  inputs:
    input_dir:
      type: Directory
      label: Image directory
  outputs:
    output_dir:
      type: Directory
      label: Output directory
      outputSource: node_run_inference/output_dir
  steps:
    node_run_inference:
      run: "#run_inference"
      in:
        input_dir: input_dir
      out:
        - output_dir

- class: CommandLineTool
  id: run_inference
  label: Flood detection model inference
  doc: Run flood detection model inference image
  requirements:
    ResourceRequirement:
      coresMin: 1
      coresMax: 2
      ramMin: 2048
      ramMax: 4096
  hints:
    DockerRequirement:
      dockerPull: eoepca/flood-detection-model:latest
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
