
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot 
import pandas as pd
import logging
import os

logging_str="[%(asctime)s:%(levelname)s: %(module)s]%(message)s"
logging_dir="logs"
os.makedirs(logging_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir,"running_logs.log"),level=logging.INFO, format=logging_str)

def main(data, eta, epochs, filename, plotfilename):    
    df = pd.DataFrame(data)
    logging.info(f"This the actual dataframe {df}" )
    X,y = prepare_data(df)  

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()
    save_model(model, filename=filename)

    save_plot(df, plotfilename,  model)

if(__name__ =='__main__'):
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    try:
        logging.info(">>>>>>>>>>>>>>> Starting Training here >>>>>>>>>>>>>>>")
        main(OR, eta=0.3, epochs=100, filename="or.model", plotfilename="or.png")
        logging.info("<<<<<<<<<<<<<< Training Done Successfullt <<<<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)    
        raise e
        