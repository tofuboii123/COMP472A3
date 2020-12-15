from NB_BOW_OV import * 
from NB_BOW_FV import * 


nb_ov = NB_BOW_OV()
nb_ov.train("./training/covid_training.tsv")

# counterYes = 0
# counterNo = 0

# for elem in nb_ov.training_tweets.values():
#     if elem == "yes":
#         counterYes += 1
#     else:
#         counterNo += 1

# print(counterYes, counterNo)

nb_ov.predict("./test/covid_test_public.tsv", "./trace/trace_NB-BOW-OV.txt", "./eval/eval_NB-BOW-OV.txt")


# counterYes = 0
# counterNo = 0

# for elem in nb_ov.test_tweets.values():
#     if elem[1] == "yes":
#         counterYes += 1
#     else:
#         counterNo += 1

# print(counterYes, counterNo)


nb_fv = NB_BOW_FV()
nb_fv.train("./training/covid_training.tsv")
nb_fv.predict("./test/covid_test_public.tsv", "./trace/trace_NB-BOW-FV.txt", "./eval/eval_NB-BOW-FV.txt")




