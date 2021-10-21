import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import sys
sys.path.append('./ProjectTools')

import ConfigHelper as cfg

import DatasetLoader as dl
import DatasetFormatter as df
import CGANKerasModel as km
import LayerDefinition as ld
import LetterProducer as lp
import CGANTrainer as ct

from tensorflow import keras

#Constants
batch_size = cfg.GetIntValue("CGAN", "BatchSize")
num_channels = 1
num_classes = cfg.GetIntValue("CGAN", "NumberOfClasses")
image_size = cfg.GetIntValue("CGAN", "ImageSize")
latent_dim = cfg.GetIntValue("CGAN", "LatentDimension")
epoch_count = cfg.GetIntValue("CGAN", "EpochCount")
refreshEachStep = cfg.GetIntValue("CGAN", "RefreshUIEachXIteration")
imageCountToProduce = cfg.GetIntValue("CGAN", "NumberOfFakeImagesToOutput")

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes

# Setup the CGAN
layerDefiniton = ld.LayerDefinition(discriminator_in_channels,generator_in_channels)

cond_gan = km.ConditionalGAN(
    discriminator=layerDefiniton.GetDiscriminator(), 
    generator=layerDefiniton.GetGenerator(), 
    latent_dim=latent_dim, 
    imageSize=image_size, 
    numberOfClasses=num_classes
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Load dataset
dataLoader = dl.DatasetLoader('../Data/Output/','',(image_size,image_size))
dataLoader.LoadTrainDatasets()
dataArray = dataLoader.DataSets

bulkDatasetFormatter = df.BulkDatasetFormatter(dataArray, num_classes,batch_size)
tensorDatasets = bulkDatasetFormatter.ProcessData();

# Train the CGAN
cGANTrainer = ct.CGANTrainer(cond_gan,tensorDatasets,epoch_count,refreshEachStep)
cGANTrainer.TrainCGAN()
trained_gen = cGANTrainer.CGAN.generator

# Use the trained generator
sentinel = True
while(sentinel):
    Question = input(f"Enter a new index to generate (0-{num_classes - 1}))(type N to exit):")
    if Question == "N":
        sentinel = False
        break

    value = int(Question)

    if value >= num_classes:
        print(f"Please write numbers within 0-{num_classes - 1}")
        continue
    if value < 0:
        print(f"Please write numbers within 0-{num_classes - 1}")
        continue

    letterProducer = lp.LetterProducer(trained_gen, num_classes, latent_dim)

    images = letterProducer.GenerateLetter(value, 10)
    letterProducer.SaveImagesAsGif(images)