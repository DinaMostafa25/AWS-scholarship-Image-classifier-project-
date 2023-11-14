import argparse
import data_utility  
import  model_functions
import json 
parser = argparse.ArgumentParser(description='Pridicting the flower class and the probability of being true')
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('--top_k')
parser.add_argument('--category_name')
parser.add_argument('--gpu', action = 'store_true')

 
args = parser.parse_args()


top_k = 1 if args.top_k is None else int(args.top_k)
category_name ='cat_to_name.json' if args.category_name is None else args.category_name
gpu = False if args.gpu is None else args.gpu

model = model_functions.load_model(args.checkpoint)
print('MODEL:\n()'.format(model))

top_p, pred_classes = model_functions.predict(data_utility.process_image(args.image_path), model, top_k)


with open(category_name, 'r') as f:
    cat_to_name = json.load(f)
classes = []
for pc in pred_classes:
    classes.append(cat_to_name[pc])
    
print(top_p)
print(classes)