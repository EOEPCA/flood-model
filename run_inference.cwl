#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool

inputs:
  input_id:
    type: string
    inputBinding:
      position: 2
  model_file:
    type: string
    inputBinding:
      position: 1
outputs:
  outputs_files:
    type: File
    outputBinding:
      glob: $(runtime.outdir)/predictions/prediction.tif

hints:
  DockerRequirement:
    dockerPull: inference