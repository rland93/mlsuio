---
title: 'Designing Target ID on a Aerial UAV System Part II'
author: Mike Sutherland
date: 2021-05-12
tags: ['uav']
---

We left off, in [Part I]({{<relref "target-id-pt1">}}), with some mission requirements and the idea to use deep learning for the imaging system. In order to build a ML model, though, we need data. For this, we can generate the targets synthetically.

Our problem lends itself to synthetic data generation. We know that the targets are 2d, which simplifies rendering significantly, becuase we don't need to worry about object modeling, light interactions, and other various complications of adding a third dimension. [1] We also know that they are flatly colored, which makes generating object surfaces extremely simple. Synthetic data generation speeds up our timeline significantly, and allows us to create a lot of data. 

Early on, I had the idea of incorporating collecting target information from test flights directly: the idea being, we put a bunch of targets on the ground, capture video, and then generate object annotations either by hand or (in the case of color-saturated objects) by their color. But, that's a long and quite a time consuming process, and further, to make all possible targets (which, by their nature, include the letter) we would need a lot of test flights. If we created 10 simple shapes, we would need 360 (10 * 36) letter:shape compositions. Creating and capturing all of those would be laborious, and using only a subset would be leaving target information "on the table" during training.

We chose synthetic dataset generation for these reasons.

# The Data Pipeline

We start with Blender. The general idea is that we can create some objects, place those objects into virtual space, fill the virtual space with data from other sources, and then render out the final image (and the annotation) from the virtual scene. Luckily, we can script Blender relatively easily with Python. [2]

We can download the blender executable, and run blender "headless," with commands given to it from a python script. This may be tricky to set up depending on your operating system and environment.

```bash
blender --background --use-extension 1 -E CYCLES -t 0 -P 'datagen.py' -- args
```

The important flag is `-P`, which tells blender to run the script `datagen.py`. We use the package [bpy](https://github.com/TylerGubala/blenderpy) to interact with the blender scene (and all of its objects) like a python object. Blender's API has full coverage, so anything we can do in Blender, we can also do with bpy. We can also pass arbitrary python arguments (after `--`)through to the script itself. We [alter argparse somewhat](https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script) to do so, to increase usability when testing different datasets under different parameters. The dataset generator I wrote can alter the camera angle and altitude, change object rotation and size (on the xy plane), so it's nice to have easy script inputs for those values.

First, we import the objects into the scene. I use a collection of shape and letter objects (14 shapes and 36 letters/numbers). The script scans for `.obj` files in a directory and imports them automatically, iterating over all letters/shapes present. That way, we generate data with every combination of shapes and letters.

While we do need every letter represented in the dataset, we only use the shapes as classes (our model doesn't need ~500 classes!). Most classification tasks benefit from having many examples of each class, and with no variation in the number of examples between classes. So we 

---

[1] If the objects were 3d, we still may have used synthetic data, either in whole or for augmentation. Objects from really high up look 2d in practice. But the rendering process would have involved me learning way more blender.

[2] I actually find that interacting with Blender through a script is *easier*, not harder. I'm sure the interface works for 3d professionals, but I never really got used to it.