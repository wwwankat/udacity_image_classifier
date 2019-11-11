
from utils import create_network
from utils import get_train_args
from utils import load_data
from utils import save_checkpoint
from utils import test_network
from utils import train_network

# sample usage: python train.py './flowers/'
def main():
    in_args = get_train_args()
    
    dataloaders, image_datasets = load_data(in_args)
    
    model = create_network(in_args)
    
    train_network(in_args, model, dataloaders)
        
    save_checkpoint(model, in_args, image_datasets) 
    
    test_network(model, dataloaders, in_args)   
    
    return None
    
  
if __name__ == "__main__":
    main()    
    