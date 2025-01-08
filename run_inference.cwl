#!/usr/bin/env cwl-runner

cwlVersion: v1.2
class: CommandLineTool

inputs:
  input_id:
    type: File
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