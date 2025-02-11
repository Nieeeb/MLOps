from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

class ThermalDataset(Dataset):
    def __init__(self, metadata_file_path, transform = None):
        self.metadata = pd.read_csv(metadata_file_path)

   
    def __len__(self): 
        return len(self.metadata)
    
    def __getitem__(self, idx):
        input_path = self.inputs[idx]
        target_path = self.targets[idx]
        input_image = Image.open(input_path)
        if not self.flare:
            input_image = F.to_tensor(input_image)
        target_image = Image.open(target_path)
        target_image = F.to_tensor(target_image)

        if self.flare: 
            _,_,input_image,_,flare=self.flare_image_loader.apply_flare(input_image)

            input_image = input_image.transpose(1, 2)
            if self.transform:
                input_image, target_image = identical_transform(self.transform, input_image, target_image)

        else:
            if self.transform:
                input_image, target_image = identical_transform(self.transform, input_image, target_image)

        return input_image, target_image

class LolTestDatasetLoader(LolDatasetLoader):
    def __init__(self, flare: bool, transform = None):
        self.flare = flare
        self.inputs = []
        self.targets = []
        self.LensFlareLowLight = False
        self.LowLightLensFlare = False
        self.inputs_dirs = [r'Data/LOLdataset/eval15/low', r"Data/LOL-v2/Real_captured/Test/Low"]
        self.targets_dirs = [r'Data/LOLdataset/eval15/high', r"Data/LOL-v2/Real_captured/Test/Normal"]
        self.transform = transform
        scattering_flare_dir=r"Data/Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare"
        self.flare_image_loader=Flare_Image_Loader(transform_base=None,transform_flare=None)
        self.flare_image_loader.load_scattering_flare('Flare7K', scattering_flare_dir)
        included_extenstions = ['png']
        self.inputs.extend(get_images(self.inputs_dirs, included_extenstions))
        self.targets.extend(get_images(self.targets_dirs, included_extenstions))