---
title: 'Designing Target ID on a Aerial UAV System, Part I'
author: Mike Sutherland
date: 2021-05-10
draft: true
tags: ['uav']
---

## Background
Ever year (except for 2020 and 2021, due to Covid-19), there is a student competition in Maryland to design and build an autonomous aerial system. The competition is put on by the Association for Unmanned Vehicle Systems International, or AUVSI. The competition is called [Student Unmanned Aerial Systems Competition (SUAS)](https://www.auvsi-suas.org/).

AUVSI puts out [a long document (PDF Warning!)](https://www.auvsi-suas.org/s/auvsi_suas-2021-rules.pdf) that details the capabilities a student vehicle must have to participate in the competition. The requirements include extensive autonomous operation, including:

+ Interoperability (Uploading telemetry autonomously to a master system)
+ Autonomous Flight (Navigating the competition space)
+ Obstacle Avoidance (Stationary obstacles, or other vehicles)
+ Object Detection (Detecting and imaging targets on the ground)
+ Mapping (Creating a map of the terrain)
+ Air Delivery (Delivery of an unmanned ground vehicle, or UGV)

Additionally, teams have to build or procure a vehicle capable of performing these tasks.

These requirements naturally require a pretty complex piece of software to be built. This project, called UAV Forge at UCI, was inititated through the mechanical engineering department. I'm in the mechanical engineering program, but I wanted to write software, so I wound up being one of only a handful of programmers working on the project.

## Object Detection

This is going to be the first of a series of posts about our object detection, localization, and classification system. In 2020, I began seriously considering the software requirements for  winning the SUAS Competition. The competition rules score teams based on their ability to find targets in the search area and identify them correctly.

An ODCL system takes raw image data as input, and gives out identifying information about targets as output. Our system answers a few questions:

+ Is there a target in the image? (*detection*)
+ If so, where is the target in the image? (*localization*)
+ What is that thing in the image? (*classification*)

Getting down to the specifics, we can look at what AUVSI demands for a fully scored target submission. They call the submission of a single target "an odlc." Each AUVSI target is a colored letter painted onto a colored shape. This is what's contained in an odlc:

+ Shape 
+ Letter 
+ Shape Color 
+ Letter Color
+ GPS Location
+ Letter Orientation
+ Cropped Image of Target

They don't give much indication as to what targets look like, but we get a few hints from a picture on the website. 

![Auvsi Targets](/img/auvsi-targets.png)

These look pretty simple, although there are some things that I notice about them that may prove to be a challenge. First, they're not all very saturated colors. There are some white, black, and grey targets pictured. Second, there are green, brown, and yellow targets, all of which will not significantly contrast with a natural background (like grass or dirt). Third, letters, while readable, are not super contrasted with their shape. That "E" pentagon and "C" semicircle look difficult. These aspects of the targets will prove to be a challenge for any computer vision system. I have to applaud the designers of the SUAS competition for crafting a difficult challenge. 

How can we tackle this problem?

We may catch some targets with clever color manipulations -- maybe we can find the most colorful blob via a [k-means](https://lmcaraig.com/color-quantization-using-k-means) algorithm?<sup>&dagger;</sup> This would catch many of the brightly colored targets, although there's risk of brightly colored false positives. It would fail to catch all the greyscale targets, though. Or perhaps we can use some kind of [edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector) to pick out target edges, then classify those. Or some combination of these techniques.

What would be best, though, would be a way of considering all of the information we know (or can guess) about a potential target, and leveraging it for detection.

## Deep Learning

Haha, yep. We're gonna throw some neural networks at the problem. Unless you've been living a prehistoric hunter-gatherer lifestyle, you are probably aware that deep learning is incredibly effective for computer vision tasks. It is a natural choice for us, because of the downsides above. Deep learning, though, is not without downsides. We're faced with a few additional considerations in comparison to a classical approach.

First is the compute power required to run inference. Our vehicle forces significant energy limitations upon us. Computers, especially those running a live video feed through a neural network in real time, eat a lot of energy. So the physical attributes of our computer system are parameters that we can tune during the design process: its weight and power draw. This affects the *temporal* resolution of our system. The less time it takes to scan an image for targets, the more chances we get at finding a target in a given period of time.

We also run into some constraints and parameters about the vehicle itself: 

+ the vehicle's altitude -- we can resolve less detail from higher up
+ the camera field of view -- we can resolve less detail the wider the camera is, but more of the field can be imaged at once with a higher FOV
+ resolution -- well, not so much the sensor resolution, but the inferencing resolution; smaller input resolutions have lower detail, but can be run faster
+ the vehicle speed -- we can get more chances at finding a target if we travel slower, because the target will be in our camera's view for longer

Some of these parameters are woven to the design of the vehicle itself. Flight time is a very hard mechanical constraint, so we may not be able to set our camera field of view super narrow, because we would need to chart a much longer course to image the same area. On the other hand, if we flew up high and got the entire field in our camera's FOV, we would probably fail to resolve any objects at the inference resolution. And we can't just slap a GTX 3090 onto the vehicle, because we can't power it (or lift it, or even get hold of one at MSRP!). But, none of these things place hard constraints on us. There are many low-power inferencing solutions out there -- smartphones, mobile TPUs, or even possibly the CPU of our on-board Raspberry Pi. I believe there is an optimal choice of components that minimizes energy use and covers the entire search area, while still imaging fast enough to catch all targets.

The other big problem is the data. Where can we find a dataset of AUVSI targets? Cars, people, animals, buildings, and so on all have open-source datasets. And even for custom objects (like the face-mask detector project that seems to have popped up everywhere) data is pretty easy to find: just search and scrape the internet. These targets are pretty custom. I'm not aware of any datasets out there or even any google searches that could lead me to an adequate dataset.

Fortunately, the targets are pretty simple. After having the conversation I've outlined in this article, we arrived at a solution: generating a synthetic dataset for our targets. I'm going to go into detail about this in the next post.

---

&dagger; This will show up later, when we examine the color of the shape.