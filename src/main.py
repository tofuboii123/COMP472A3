from NB_BOW_OV import * 
from NB_BOW_FV import * 


nb_ov = NB_BOW_OV()
nb_ov.train("./training/covid_training.tsv")
nb_ov.predict("./test/covid_test_public.tsv", "./trace/trace_NB-BOW-OV.txt", "./eval/eval_NB-BOW-OV.txt")


nb_fv = NB_BOW_FV()
nb_fv.train("./training/covid_training.tsv")
nb_fv.predict("./test/covid_test_public.tsv", "./trace/trace_NB-BOW-FV.txt", "./eval/eval_NB-BOW-FV.txt")




