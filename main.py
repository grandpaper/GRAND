import pandas as pd
import torch
import pandas
from models import ADNN
from models import Parameters
from models import Trainer

if __name__ == "__main__":
    p = Parameters.Parameters()
    d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # replace the folder path in the trainer
    trainer = Trainer.Trainer("/home/ksy/data/norm-total/norm", p, device=d)
    trainer.train()