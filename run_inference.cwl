cwlVersion: v1.0
$graph:
- class: Workflow
  id: main
  label: Run inference
  doc: Run an inference for flood detection
  inputs:
    input_id:
      label: Input Id
      type: File 
  outputs:
    output_file:
      label: Output File
      type: File
      outputSource: node_run_workflow_inference/output_file 

  steps:
    node_run_workflow_inference:
      run: "#run_workflow_inference"
      in:
        input_id:
          source: input_id
      out: 
        - output_file 

- class: CommandLineTool
  id: run_workflow_inference
  label: Run inference tool
  doc: Run an inference for flood detection
  hints:
    SoftwareRequirement:
      packages:
        - package: flood-inference
          version: ["latest"]
    DockerRequirement:
      dockerPull: inference
  inputs:
    input_id:
      type: File
      inputBinding:
        position: 1
  outputs:
    output_file:
      type: File
      outputBinding:
        glob: "predictions/prediction.tif"