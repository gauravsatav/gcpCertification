import os

foldersNames = ['0_A_Tour_of_Google_Cloud_Hands-on_Labs','1_Google_Cloud_Big_Data_and_Machine_Learning_Fundamentals','2_How_Google_Does_Machine_Learning','3_Launching_into_Machine_Learning','4_Introduction_to_TensorFlow','5_Feature_Engineering','6_Art_and_Science_of_Machine_Learning','7_Production_Machine_Learning_Systems','8_Image_Understanding_with_TensorFlow_on_Google_Cloud','9_Sequence_Models_for_Time_Series_and_Natural_Language_Processing_on_Google_Cloud','10_Recommendation_Systems_with_TensorFlow_on_Google_Cloud','11_MLOps_(Machine_Learning_Operations)_Fundamentals','12_ML_Pipelines_on_Google_Cloud','13_Perform_Foundational_Data,_ML,_and_AI_Tasks_in_Google_Cloud','14_Explore_Machine_Learning_Models_with_Explainable_AI','15_Build_and_Deploy_Machine_Learning_Solutions_on_Vertex_AI']

for f in foldersNames:
    os.makedirs(f)



for f in foldersNames:
    try:
        with open(os.path.join(f,'notes.md'),'w+') as file:
            file.write(f'''
|[Home](../README.md)|[Course Page]()|
|---------------------|--------------|

## {f.replace('_',' ')}

[TOC]
        ''')
    except:
        pass
