---
title: "Side Project Ideas"
---
Legend:
+ ☆☆☆☆☆ Just thought of it
+ ★☆☆☆☆ Researched it a bit to see what it'd take to do it
+ ★★☆☆☆ Started working on it
+ ★★★☆☆ Made a hacky prototype
+ ★★★★☆ Made a pretty good prototype
+ ★★★★★ Completed and hosted/implemented

## ★☆☆☆☆ Generate Regex Examples

This could be a simple site. There's a python package, https://pypi.org/project/xeger/. It can generate random strings based on a regex. The idea is to have the user type a regex in and then get dozens, hundreds, or thousands of strings back (in an easily copyable form). 

Other features would be to generate slightly-failing strings or majorly failing strings from the supplied regex pattern. 

## ★★★☆☆ CoursePlanner

This is a University of California, Irvine specific thing. 

I want to finish my degree in 3.5 years. I have a full graph of the courses, including their prerequisites, corequisites, unit count, etc. I can even scrape data from professors, like the grade point average of the course, their rating, and so on. I can also find the teaching plan every year and put it into the tool so that I can guarantee classes are offered in a given quarter.

Then, it's a simple graph traversal to get all prereqs and coreqs to take a given set of classes. You could use it to plan courses outside your major, or minor courses.

## ★★☆☆☆ Polynomial Glider

You have a simple unpowered plane, or a rubber band plane. You have some set of points. What if you wanted to compute a smooth flight plan through the points, and then throw the plane and have it glide through the course you set? 