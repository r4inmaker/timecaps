# IMPORTS

import configparser


# PARSER

def get_args():
    
    parser = {}
    config = configparser.ConfigParser()
    config.read('config.ini') 
    
    # DATA
    parser['mitbih-preprocessed-path'] = config.get('DATA', 'MIT-BIH-PREPROCESSED-PATH')
    parser['url'] = config.get('DATA', 'URL')
    parser['mitbih-path'] = 'mit-bih-arrhythmia-database-1.0.0'
    
    # MODEL
    parser['batch-size'] = int(config.get('MODEL', 'BATCH_SIZE'))
    parser['epochs'] = int(config.get('MODEL', 'EPOCHS'))
    parser['model-name-save'] = config.get('MODEL', 'MODEL_NAME_SAVE')
    parser['model-name-load'] = config.get('MODEL', 'MODEL_NAME_LOAD')
    parser['L'] = int(config.get('MODEL', 'L'))
    parser['K'] = int(config.get('MODEL', 'K'))
    parser['g1'] = int(config.get('MODEL','g1'))
    parser['cp'] = int(config.get('MODEL', 'cp'))
    parser['ap'] = int(config.get('MODEL', 'ap'))
    parser['g2'] = int(config.get('MODEL', 'g2'))
    parser['n'] = int(config.get('MODEL', 'n'))
    parser['cSA'] = int(config.get('MODEL', 'cSA'))
    parser['aSA'] = int(config.get('MODEL', 'aSA'))
    parser['g3'] = int(config.get('MODEL', 'g3'))
    parser['cb'] = int(config.get('MODEL', 'cb'))
    parser['ab'] = int(config.get('MODEL', 'ab'))
    parser['cSB'] = int(config.get('MODEL', 'cSB'))
    parser['aSB'] = int(config.get('MODEL', 'aSB'))
    parser['gB'] = int(config.get('MODEL', 'gB'))
    parser['n_classes'] = int(config.get('MODEL', 'n_classes'))
    
    return parser

