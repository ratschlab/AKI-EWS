''' A script to check coherence of LGBM, LSTM rankings with the Physician rankigns'''

import os
import os.path
import argparse

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def execute(configs):

    # Build curve for LGBM
    x_arr=np.arange(1,29,1)

    y_lgbm=[]
    y_lstm=[]

    for k in x_arr:

        if configs["lgbm_ranking_type"]=="corrupt":
            lgbm_prefix=configs["LGBM_CORRUPT_RANKINGS"][:k]
        elif configs["lgbm_ranking_type"]=="shap":
            lgbm_prefix=configs["LGBM_SHAP_RANKINGS"][:k]            
            
        lstm_prefix=configs["LSTM_RANKINGS"][:k]

        if configs["aggregate"]=="mean":
            lgbm_agg=np.mean([configs["PHYSICIAN_RANKINGS"][k] for k in lgbm_prefix])
            lstm_agg=np.mean([configs["PHYSICIAN_RANKINGS"][k] for k in lstm_prefix])
        elif configs["aggregate"]=="sum":
            lgbm_agg=sum([configs["PHYSICIAN_RANKINGS"][k] for k in lgbm_prefix])
            lstm_agg=sum([configs["PHYSICIAN_RANKINGS"][k] for k in lstm_prefix])            
        
        y_lgbm.append(lgbm_agg)
        y_lstm.append(lstm_agg)

    plt.plot(x_arr,y_lstm,label="LSTM")
    plt.plot(x_arr,y_lgbm,label="GBDT-snapshot")
    plt.xlabel("k most important variables")
    plt.ylabel("Relevance of variable set (physician-rated), {}".format(configs["aggregate"]))
    plt.legend()    
        
    plt.savefig(os.path.join(configs["plot_path"],"physician_agreement_{}.pdf".format(configs["aggregate"])))
    plt.savefig(os.path.join(configs["plot_path"],"physician_agreement_{}.png".format(configs["aggregate"])),dpi=300)

    

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths

    # Output paths
    parser.add_argument("--plot_path",default="../../data/plots/introspection",
                        help="Plotting folder")

    # Arguments
    parser.add_argument("--aggregate", default="mean", help="Aggregate variable importance")

    parser.add_argument("--lgbm_ranking_type", default="corrupt", help="Ranking to use for LGBM")

    configs=vars(parser.parse_args())

    configs["PHYSICIAN_RANKINGS"]={ "plain_vm275": 3, # Dialysis
                                    "plain_pm93": 1, # Enteral feeding
                                    "plain_pm290": 1, # Laxatives
                                    "plain_vm276": 3, # OUTurine/h
                                    "plain_vm24": 3, # OUTurine/c
                                    "plain_vm131": 2, # Weight
                                    "plain_vm156": 3, # Creatinine
                                    "plain_vm21": 1, # EtCO2
                                    "plain_vm176": 2, # C-reactive protein
                                    "plain_pm94": 1, # Parenteral feeding
                                    "plain_pm69": 2.5, # Loop diuretics
                                    "plain_pm92": 1, # Thrombozytenhemmer
                                    "plain_pm95": 1.5, # Heparin
                                    "plain_pm35": 2, # Packed RBC
                                    "plain_pm43": 2, # Levosimendan
                                    "plain_vm154": 1.5, # Magnesium (lab)
                                    "plain_vm65": 1, # RRset
                                    "plain_vm31": 2.5, # IN (fluid intake)
                                    "plain_vm313": 1, # Tracheotomy state
                                    "plain_vm162": 1.5, # Bilirubin total
                                    "plain_pm109": 1, # Anti-delirant medication
                                    "plain_pm86": 1, # Opiate
                                    "plain_vm226": 1, # Sekretmenge
                                    "plain_pm73": 2, # Antibiotics
                                    "plain_pm101": 1.5, # Kalium (Potassium) drug
                                    "plain_pm104": 1.5, # Mg (drug)
                                    "RelDatetime": 2,
                                    "Emergency": 2}

    configs["LSTM_RANKINGS"]=['plain_vm275', 'plain_pm93', 'plain_vm276', 'plain_vm176', 'plain_vm21', 'plain_vm156', 'plain_pm290', 'plain_pm94', 'plain_pm92',
                              'plain_vm31', 'plain_vm65', 'plain_vm313', 'plain_pm95', 'plain_pm43', 'plain_vm154', 'plain_pm104', 'plain_pm86', 'plain_vm131',
                              'plain_vm24', 'plain_vm162', 'Emergency', 'plain_pm35', 'plain_vm226', 'RelDatetime', 'plain_pm69', 'plain_pm73', 'plain_pm109', 'plain_pm101']

    configs["LGBM_CORRUPT_RANKINGS"]=['RelDatetime', 'plain_vm131', 'plain_pm93', 'plain_pm290', 'plain_vm24', 'plain_pm69', 'plain_pm109', 'plain_pm35', 'plain_vm156',
                                      'plain_pm95', 'plain_vm276', 'plain_pm104', 'plain_pm73', 'plain_vm275', 'plain_vm21', 'plain_pm43', 'Emergency', 'plain_vm176', 'plain_pm101',
                                      'plain_vm162', 'plain_vm65', 'plain_vm226', 'plain_pm86', 'plain_vm313', 'plain_vm31', 'plain_pm92', 'plain_vm154', 'plain_pm94']

    configs["LGBM_SHAP_RANKINGS"]=["RelDatetime","plain_pm93","plain_pm69","plain_vm131","plain_pm109",
                                   "plain_pm290","plain_pm104","plain_vm24","plain_pm35","plain_pm95",
                                   "Emergency","plain_pm73","plain_pm101","plain_vm156","plain_vm276",
                                   "plain_vm21","plain_pm86","plain_vm275","plain_vm65","plain_vm162",
                                   "plain_pm43","plain_vm226","plain_pm92","plain_pm94","plain_vm154","plain_vm131"]
    
    execute(configs)
