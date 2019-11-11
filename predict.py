
from utils import get_predict_args
from utils import load_checkpoint
from utils import predict

# sample usage: python predict.py './flowers/test/1/image_06764.jpg' './checkpoint.pth'
def main():
    in_args = get_predict_args()
    
    model = load_checkpoint(in_args)
    
    predict(model, in_args)
    
    return None

if __name__ == "__main__":
    main()