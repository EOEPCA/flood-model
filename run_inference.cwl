cwlVersion: v1.2
class: CommandLineTool

hints:
    DockerRequirement:
        dockerPull: inference

inputs:
    model_file:
        type: string
        inputBinding:
            position: 1

    input_id:
        type: string
        inputBinding:
            position: 2

outputs:
    outputs_files:
        type: File
        outputBinding:
            glob: "predictions/prediction.tif"

