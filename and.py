    """
    author: Namrata Choudhary
    email: choudharynamu90@gmail.com 
    """
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot 
import pandas as pd

def main(data, eta, epochs, filename, plotfilename):    
    df = pd.DataFrame(data)
    print(df )
    X,y = prepare_data(df)  

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()
    save_model(model, filename=filename)

    save_plot(df, plotfilename,  model)

if(__name__ =='__main__'):
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    main(AND, eta=0.3, epochs=100, filename="and.model", plotfilename="and.png")