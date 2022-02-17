---
title: 'UAV Target ID: Data Generation'
author: Mike Sutherland
date: 2021-07-20
draft: True
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

The important flag is `-P`, which tells blender to run the script `datagen.py`. We use the package [bpy](https://github.com/TylerGubala/blenderpy) to interact with the blender scene (and all of its objects) like a python object. Blender's API has full coverage, so anything we can do in Blender, we can also do with bpy. We can also pass arbitrary python arguments (after `--`)through to the script itself. We [alter argparse somewhat](https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script) to do so, to increase usability when testing different datasets under different parameters. We can alter the camera angle and altitude, change object rotation and size (on the xy plane), etc. so it's nice to have easy script inputs for those values.

First, we have our objects. I use a collection of shape and letter objects (14 shapes and 36 letters/numbers). The script scans for `.obj` files in a directory and imports them automatically, iterating over all letters/shapes present. That way, we generate data with every combination of shapes and letters.

## Side Note: How Many Examples?

While we do need every letter represented in the dataset, we only use the shapes as classes (our model doesn't need (no. shapes) $\times$ (26 letters + 10 digits) classes!). Most classification tasks benefit from having many examples of each class, and with no variation in the number of examples between classes.

Although in theory, training on more examples will improve generalization; but in practice, there are a few factors that limit this. For one, we are generating our own dataset. Although we do it in a fairly "random" way -- or, a way that we hope matches with the data our object detector will eventually see in the wild -- we will still encode our own biases. More biased examples will not eliminate those biases! Secondly, there is a diminishing return on data in this case. Because our object detector is eventually going onto an embedded computer, it will be quite small in terms of the number of parameters. 

The way to think about any deep neural network is that it's just a function approximator. A really, really complex function approximator, but nonetheless, just a function approximator. A "perfect" object detector will take as inputs the raw image data, and spit out some bounding boxes and classes; to recognize any object, we need a function with basically "infinite" complexity.

The network itself is a coarse approximation. Much better than what we could craft by hand, but nonetheless: an approximation. Importantly, to do inference at low power, we are physically limited by the amount of electricity we can use to flip those bits. Making a really fine approximation of our perfect function requires us to store a lot of stuff in memory, and hence takes up power. Power that is scarce and valuable on a flying machine!

Then, we can train on a lot of examples, but after a certain point, it doesn't really matter: we won't really be able to use the extra information we gained by looking at one additional training example -- it will be lost when our model is quantized.

So, just make a lot of training examples, but not so much that it takes forever to train. I'd say about 1,000 per class is just fine.

## Rendering, Step by Step 

Let's step through the rendering logic.

First, we clear all the objects in the scene. Blender has these things called scenes. Adding objects into a scene causes them to be loaded into memory so that they can be rendered. So the first step is to clear whatever is in the scene.

Next, we need a source of light. We can add everything we want, but putting objects into an empty scene is like putting them in blank space. No light, no good render. Blender gives us a special type of light, called a "Sun." Suns in blender have the same power of illumination no matter how far away they are placed. So we just place it anywhere in a 1000 x 1000 meter cube centered at our current position, with a height of 1000 meters.

Now it's time to place the camera. We put a camera in and specify some random pitch/roll/yaw. The roll and yaw doesn't matter much, but we keep the pitch somewhat less than 90 degrees (in our frame of reference, 0 degrees is pointed at the ground). This is because the length of a ray line from the camera to its intersection with the horizontal ground is proportional to roughly tangent of the pitch angle. 

![you thought you could avoid trig?](img/trig.png)

Remember, SOH-CAH-TOA? TOA: opposite over ajacent; as we increase our pitch angle, our triangle's hypotenuse (the ray line length) gets larger and larger. Since the ajacent is just the camera's altitude, the opposite must increase. It blows up to infinity when we hit 90 degrees pitch angle. So let's keep it around 60 degrees. 

When we script blender, it's exactly as if we are clicking the buttons. It's just that the buttons are all clicked by our software through [an API](https://docs.blender.org/api/current/info_overview.html). So, the workflow is slightly different. For any object, we place it in the scene first, and then we access its parameters (such as position, type of object, materials, lighting, and so on) through the API. When we've changed the parameters to our liking, we deselect the object. It's not quite intuitive at first, but if we just remember that our python script is basically clicking buttons in order in Blender -- that makes it easier to understand.

Now, we have a light and a camera placed into our scene. Time to place the objects. To make training more efficient, we place multiple objects onto the scene in each frame. But, we still want to be as physically realistic as possible. How many objects should we place?

## A Little Bit of Probability

This is really a probability question in disguise. We are going to choose a random number of objects to add into the scene, so let's have a look at the physical interpretation of some random distributions, courtesy this [Wikipedia Article](https://en.wikipedia.org/wiki/Probability_distribution#Common_probability_distributions_and_their_applications):

+ **Linear growth (e.g. errors, offsets)**
+ **Exponential growth (e.g. prices, incomes, populations)**
+ **Uniformly distributed quantities**

We have neither one of these. First, they're not discrete; we need an integer number of objects to place in scene. Or, more accurately, we need some realistic distribution in the number of discrete objects in our scene.

+ **Bernoulli trials (yes/no events, with a given probability)**

This is a bit closer. But we don't have a "yes/no" event here. We're only considering those times that at least one object is present (because if it's not, that's not useful training information!). So this isn't really an event like *object is present/not present.* It describes a situation: *the plane is passing overhead. What is the likelihood that $n$ objects are spotted, for any possible $n>0$ objects?*

+ **Categorical outcomes (events with K {\displaystyle K} K + possible outcomes)**

Well, this is a bit closer to what we need. But still not quite there. Consider: *the plane is passing overhead. What is the likelihood that $n$ objects are spotted, for any possible $n>0$ objects?* We could say, there's $n$ different categories: spotting $n$ number of objects in a single frame. But are these truly categories? There is no categorical distinction between a frame with a single target and a frame with two targets; both arise from the same mechanism. So although categorical outcomes are discrete, they're not the physical interpretation we are looking for.

+ **Poisson process (events that occur independently with a given rate)**

A promising distribution is this one: 

[Poisson distribution, for the number of occurrences of a Poisson-type event in a given period of time](https://en.wikipedia.org/wiki/Poisson_distribution)

This seems to be right. We can assume that targets occur independently; they could be placed anywhere. What is the rate? Let's assume that the event which occurs is that an object appears within the frame. If we slide the frame over a certain area in which objects are scattered, objects will appear on and disappear from the frame at some rate; if they're *independently* scattered, those events will occur *independently*. 

Wikipedia says, of this distribution:

> For instance, a call center receives an average of 180 calls per hour, 24 hours a day. The calls are independent; receiving one does not change the probability of when the next one will arrive. The number of calls received during any minute has a Poisson probability distribution: the most likely numbers are 2 and 3 but 1 and 4 are also likely and there is a small probability of it being as low as zero and a very small probability it could be 10.

This sounds just like what we have here. The article notes that events do not need to have intervals of time; they could have spatial intervals instead. Poisson it is!

One more thing. Since we don't want images not containing any objects, we are really looking for a [Zero Truncated Poisson Distribution.](https://en.wikipedia.org/wiki/Zero-truncated_Poisson_distribution)

So we create a counter for each rendered image: place $n$ objects, where $n$ is the ZTP distribution. Then, just iterate through that counter with the object placement logic.

## Placing Objects




---

[1] If the objects were 3d, we still may have used synthetic data, either in whole or for augmentation. Objects from really high up look 2d in practice. But the rendering process would have involved me learning way more blender.

[2] I actually find that interacting with Blender through a script is *easier*, not harder. I'm sure the interface works for 3d professionals, but I never really got used to it.