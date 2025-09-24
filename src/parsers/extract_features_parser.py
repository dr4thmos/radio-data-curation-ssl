import argparse
import os

def parse_model_parameters(param_list):
    """Parsa i parametri del modello da una lista di stringhe key=value."""
    return {kv.split('=')[0]: kv.split('=')[1] for kv in param_list if '=' in kv}

def get_parser():
    parser = argparse.ArgumentParser(description="Parametrized script for feature extraction.")
    
    parser.add_argument('--source_id',     type=str, required=True,
                        help="which cutouts or combination of cutouts to use")
    parser.add_argument('--cuda_devices',       type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "4"),
                        help="CUDA device(s) to use in the format \"0,2,..\"")
    parser.add_argument('--features_folder',    type=str, default=os.path.join(os.getcwd(), 'features'),
                        help="Path to features folder.")
    parser.add_argument('--ckpt_path',          type=str, default=None, 
                        help="se scelto ssl, weight path")
    parser.add_argument('--model_type',         type=str, default="thingsvision", 
                        help="Model type like cecconellossl, thingsvision, ...")
    parser.add_argument('--model_name',         type=str, default=None, 
                        help="Model name like OpenCLIP, resnet18, ...")
    parser.add_argument('--model_parameters',   type=str, nargs='*', default=[], # e se non viene specificato? gestire questo caso
                        help="Model parameters as key=value pairs.")
    parser.add_argument('--backend',            type=str, default='pt', 
                        help="Backend to use.")
    parser.add_argument('--source',             type=str, default='custom', 
                        help="Data source type.")
    parser.add_argument('--batch_size',         type=int, default=16, 
                        help="Batch size for processing.")
    
    parser.add_argument('--module_name',        type=str, default=None, 
                        help="Module where to extract features: visual, avgpool...")
    parser.add_argument('--normalization',      type=str, default='imagenet', 
                        help="which norm to use")
    parser.add_argument('--resize',             type=int, default=224, 
                        help="Image dimensional resize.")
    parser.add_argument('--output_file_format', type=str, default="npy", 
                        help="npy, txt, mat, pt, hdf5")
    parser.add_argument('--test_mode', action='store_true', help='Esegui in modalità test')
    parser.add_argument('--test_batches', type=int, default=5, help='Numero di batch da processare in modalità test')
    parser.add_argument(
        '--model-input-channels',
        type=int,
        default=3,
        choices=[1, 3],
        help="Numero di canali che il modello si aspetta in input. "
             "Standard ResNet/CLIP/etc. richiedono 3. "
             "Alcuni modelli custom o DINO possono usare 1. Default: 3"
    )


    args = parser.parse_args()

    args.model_parameters = parse_model_parameters(args.model_parameters)
    return args
"""
config = {
    "model": "clip",
    "model_params": {
        "variant": "ViT-g-14",
        "dataset": "laion2b_s34b_b88k"
    },
    "backend": "pt",
    "source": "custom",
    "data_path": "LoTSS/cutouts",
    "source_id": "885f3971188c47d58139219a0598b258",
    "module_name": "avgpool",
    "normalization": "imagenet",
    "resize": 224,
    "merged_cutouts_id": "merged_id",
    "ckpt_path": ""
}
"""