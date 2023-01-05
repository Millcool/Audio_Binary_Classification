from config import *
from dataset import *
from transforms import *

def prepare_loaders(data):
    audio_names = pd.read_csv(CFG.csv_path, sep='\t')
    audio_names = list(audio_names['path'])
    # audio_names= [item_name.split('.')[0] for item_name in os.listdir(CFG.train_path)] #[os.path.join("./train/",item_name)  for item_name in os.listdir(CFG.train_path)]
    random.shuffle(audio_names)
    audio_names = audio_names[:CFG.num_item_all]
    print(f'Number of items we work with:{len(audio_names)}')
    audio_train_valid, audio_test = train_test_split(audio_names, test_size=0.2, random_state=CFG.seed)
    audio_train, audio_valid = train_test_split(audio_train_valid, test_size=0.25, random_state=CFG.seed)
    # print(type(train_ids))
    # print(valid_ids)
    train_dataset = UrbanSoundDataset(data, audio_train, MFCC_spectrogram, CFG.SAMPLE_RATE, CFG.NUM_SAMPLES, True)
    valid_dataset = UrbanSoundDataset(data, audio_valid, MFCC_spectrogram, CFG.SAMPLE_RATE, CFG.NUM_SAMPLES, True)
    test_dataset = UrbanSoundDataset(data, audio_test, MFCC_spectrogram, CFG.SAMPLE_RATE, CFG.NUM_SAMPLES, True)

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)
    return train_loader, valid_loader, test_loader
