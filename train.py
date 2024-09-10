# IMPORTS
from data import *
from model import *
from argsparser import get_args


# PARSER ARGUMENTS
parser = get_args()
# _data
URL = parser['url']
WRITE_PATH = parser['mitbih-preprocessed-path']
ROOT_PATH = parser['mitbih-path']
# _model
BATCH_SIZE = parser['batch-size']
EPOCHS = parser['epochs']
MODEL_NAME_SAVE = parser['model-name-save']
MODEL_NAME_LOAD = parser['model-name-load']
_L = parser['L']
_K = parser['K']
_g1 = parser['g1']
_cp = parser['cp']
_ap = parser['ap']
_g2 = parser['g2']
_n = parser['n']
_cSA = parser['cSA']
_aSA = parser['aSA']
_g3 = parser['g3']
_cb = parser['cb']
_ab = parser['ab']
_cSB = parser['cSB']
_aSB = parser['aSB']
_gB = parser['gB']
_n_classes = parser['n_classes']



# MODEL (train)

MODEL = TimeCaps( L=_L, K=_K, g1=_g1, cp=_cp, ap=_ap, g2=_g2, n=_n, cSA=_cSA, aSA=_aSA, g3=_g3, cb=_cb,
                  ab=_ab, cSB=_cSB, aSB=_aSB, gB=_gB, n_classes=_n_classes, 
                 batch_size=BATCH_SIZE, device=DEVICE
)

DECODER = Decoder(16)


# TRAINING LOOP

def train():
    
    '''
    Returns a trained Chef() instance after training.
    '''
    
    _, _, TRAINLOADER, TESTLOADER = get_data()

    Gordon = Chef(batch_size=BATCH_SIZE, model=MODEL, decoder=DECODER,
                  trainloader=TRAINLOADER, testloader=TESTLOADER)
    Gordon.cook(epochs=EPOCHS, model_name=MODEL_NAME_SAVE)

    
    return Gordon