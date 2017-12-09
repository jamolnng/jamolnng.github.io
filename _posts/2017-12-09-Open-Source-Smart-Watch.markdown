---
title: "Open Source Smart Watch"
layout: post
date: 2017-12-09
headerImage: false
tag: project
category: project
hidden: true
projects: true
author: jesselaning
description: Open Source Smart Watch
---

The original inspiration for this project came from Jared Sanson here [http://jared.geek.nz/2014/jul/oshw-oled-watch](http://jared.geek.nz/2014/jul/oshw-oled-watch)

This is my attempt at creating a "smart" watch, however it will only be as smart as I make it so we'll see how well I do.

You can follow the project's Github page [here](https://github.com/jamolnng/Smart-Watch)

The processor I chose is in the PIC32MZ EF family, specifically the [PIC32MZ2048EFG064](http://www.microchip.com/wwwproducts/en/PIC32MZ2048EFG064)
I had several reasons for choosing it. A big one was its large, for a microcontroller, amount of program memory and RAM. My goal is to be able to use [MicroPhyton](https://micropython.org/) to write apps but I'm not 100% sure I will be able to without adding a RAM chip. I also chose it because it can act as a USB device such as a mass storage device that I can just plug into my computer and transfer the python scrips over. The chip also has a parallel master port which makes it easy to control a compatable LCD. There are plenty of great things about this chip, its a beast. It's price is pretty large though at about $11 USD but since this is a one off design I'm not too concerned.

# Planned Features

* 1.8 inch 128x160 RGB LCD
* Real time clock (it's built into the main processor)
* Bluetooth communication with phone, hopefully sync notifications and be able to answer calls however there are no plans for a mic on the watch itself
* USB HID/Mass storage communication for easy programming of apps
* 400-500mAh battery with 500mA charging circuit
* Accelerometer
* Magnemometer
* Single/Double tap sensor for tap to wake

# Hardware
This is the current schematic, I'm finalizing the design and trying to make sure I haven't made any mistake (but I probably have)

![Schematic](https://raw.githubusercontent.com/jamolnng/Smart-Watch/master/Hardware/Schematics/SmartWatchSchematic.png)