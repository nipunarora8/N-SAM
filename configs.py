import argparse

parser = argparse.ArgumentParser(description='training argument values')

def add_training_parser(parser):
    parser.add_argument("-model_type", type=str, default="vit_h") #Choose between vit_b, vit_l, vit_h (recommended)
    parser.add_argument("-customloss", type=bool, default=False) # True and False
    parser.add_argument("-device", type=str, default="0") 
    parser.add_argument("-epochs", type=int, default=100) # Number of epochs
    parser.add_argument("-lr", type=float, default=1e-4) # Minimum Learning Rate
    parser.add_argument("-check_interval", type=int, default=5)
    parser.add_argument("-logger", type=bool, default=True)
    parser.add_argument("-trainpath", type=str, default= "datasets/DeepD3_Training")
    parser.add_argument("-valpath", type=str, default= "datasets/DeepD3_Validation")
    parser.add_argument("-testdata", type=str, default = "DeepD3_Benchmark/DeepD3_Benchmark.tif")
    parser.add_argument("-swc_filename", type=str, default = "Dendrite_U.swc") # Keep it '' if you don't have masks
    parser.add_argument("-checkpoint", type=str, default = "results/best_weights.pth")