import os
import argparse
import json

from __context__ import src
from src import calibrate
from src import dataset


def calibrateFromInputPath(inputPath):
    filePathNoExt, ext = os.path.splitext(inputPath)
    if not ext.endswith("json"):
        raise ValueError("Expected .json file path")

    allDetections = dataset.createDetectionsFromPath(inputPath)
    Ainitial, Winitial, kInitial = calibrate.estimateCalibrationParameters(allDetections)
    Arefined, Wrefined, kRefined = calibrate.refineCalibrationParametersSciPy(
            Ainitial, Winitial, kInitial, allDetections)

    outputDict = {
        "boardPoses": [cMw.tolist() for cMw in Wrefined],
        "intrinsics": Arefined.tolist(),
        "distortion": kRefined,
    }
    outputPath = f"{filePathNoExt}.output.json"
    with open(outputPath, "w") as f:
        f.write(json.dumps(outputDict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=("Produces a camera calibration for an input file"))
    parser.add_argument(
        "inputPath",
        help="Path to input file (.json)"
    )
    args = parser.parse_args()
    calibrateFromInputPath(args.inputPath)
