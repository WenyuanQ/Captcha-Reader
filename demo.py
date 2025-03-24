import argparse
from captcha import Captcha
from model import CaptchaModel
from captcha_simple import CaptchaSimple

parser = argparse.ArgumentParser(description="Run captcha solver")
parser.add_argument('im_path', type=str, help="Path to the captcha image")
parser.add_argument('--model', choices=['simple', 'neural'], default='simple', help="Choose the captcha model to use ('simple' or 'neural')")
parser.add_argument('--model_path', type=str, default='captcha_model.pth', help="Model checkpoint, default 'captcha_model.pth' (only needed for 'neural' model)")
parser.add_argument('--save', type=str, default=None, help="Path to save the result (optional)")

args = parser.parse_args()

def run_captcha_simple(im_path, save_path=False):
    """
    Run the CaptchaSimple model.
    """
    captcha = CaptchaSimple()
    return captcha(im_path, save_path)

def run_captcha(im_path, model_path, save_path=False):
    """
    Run the Captcha neural network model.
    """
    model = CaptchaModel()
    captcha = Captcha(model, model_path)
    return captcha(im_path, save_path)

def main():
    if args.model == 'simple':
        result = run_captcha_simple(args.im_path, save_path=args.save)
    elif args.model == 'neural':
        if not args.model_path:
            print("Error: --model_path is required when using 'neural' model.")
            return
        result = run_captcha(args.im_path, args.model_path, save_path=args.save)
    
    print(f"Predicted Result: {result}")

if __name__ == "__main__":
    main()