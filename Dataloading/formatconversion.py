import pandas as pd
import json
from tqdm import tqdm

def main():
    val_file = r'/home/nieb/Projects/Big Data/Images/Seasons_drift/v2/harborfrontv2/Train.json'
    images_df = pd.DataFrame()
    annotations_df = pd.DataFrame
    with open(val_file) as json_data:
        data = json.load(json_data)
        images_df = pd.DataFrame(data['images'])
        annotations_df = pd.DataFrame(data['annotations'])
        json_data.close()
    print(len(annotations_df))
    print(images_df.columns)
    print(annotations_df.columns)

    with tqdm(total=images_df.shape[0]) as pbar:
        annotation_index = 1
        patience = 5
        for index1, image in images_df.iterrows():
            pbar.update(1)
            matches = []
            items_since_match = 0
            for index2 in range(annotation_index -1, len(annotations_df)):
                if items_since_match < patience:
                    if image['id'] == annotations_df.iloc[index2]['image_id']:
                        matches.append(annotations_df.iloc[index2])
                        annotation_index += 1
                        items_since_match = 0
                    else:
                        items_since_match += 1
                else:
                    items_since_match = 0
                    break
            image['annotations'] = matches
            #print(f"Found {len(image['annotations'])} in image {image['id']}")


if __name__ == '__main__':
    main()