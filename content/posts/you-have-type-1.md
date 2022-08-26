---
title: "You Have Type 1, Part One"
date: 2022-08-22
tags: ['You Have Type 1']
draft: False
---
So, I have been working on this project. It's called You Have Type 1. It's a game where you get diagnosed with Type 1 diabetes, and then you have to manage your diabetes for a week. My goal for the game is to basically make it really difficult to manage your diabetes -- and give non-diabetics an intuition for what it's like to have and manage Type 1.

This game uses the [UVA/Padova Type 1 Diabetes Simulator](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454102/) to simulate the player as a patient. This simulator is quite accurate and is tuned to real-world patients. [A free python package of the 2008 simulator](https://github.com/jxx123/simglucose) was released -- it's called "simglucose." People are using it to do reinforcement learning.

I ripped out the guts of this simulator and am building a web interface to it. The idea is pretty simple but it offers me a good opportunity to learn how web-apps (at least, server-side) are built (and hopefully, scaled!). 

The game is pretty simple: you have to manage your hunger and your blood glucose. If you get too hungry, you lose, which forces you to eat. But if you eat, you have to give insulin. So you have to determine, just like a real diabetic, how much insulin you need to give. If you dose your insulin wrong, you'll get hypoglycemia or hyperglycemia, which you then have to manage.

You'll do this for 7 simulated days, and then at the end of the "week," you're scored on your performance using [clinically relevant risk indicators](https://www.tandfonline.com/doi/abs/10.1080/10273660008833060). At the end, you'll see things that [researchers and clinicians use](https://ieeexplore.ieee.org/abstract/document/5342789?casa_token=1A0tgm_JuC4AAAAA:bqSyU6bgGe90Nrif02eCBIJ-czEO2d_0TTMwH4aUDHFplzpkAMpQ6zPW05dmU02wNVPYtvZG0u0) to assess diabetes care, like a poincare plot, CVGA plot, and risk index plot.

The lower your cumulative score, the better you did! I eventually plan on having the ability to share your game summary with others. I plan on making a bunch of "hard modes" that mimic the things that make real diabetes a challenge: you forget to give insulin; you have incomplete information about your blood glucose (like needing to proactively test your blood sugar); you get sick, stressed, or need to exercise, which puts an external strain on your sugars; and so on. 

So far, this is what the game looks like:

<img class="pct70 centered" src="/img/t1d_img/t1d-game-1.png">

My intention is to make the interface look like a CGM / pump system. Hopefully, by playing this game, people can get a little bit of intuition about how F!@($*)ing difficult it is to manage Type 1 Diabetes.