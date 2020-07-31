/*
 *  FrameCenter.cpp
 *  samurai
 *
 *  Copyright 2008 Michael Bell. All rights reserved.
 *
 */

#include "FrameCenter.h"

FrameCenter::FrameCenter()
{
	latitude = -999;
	longitude = -999;
	Um = -999;
	Vm = -999;
	char const* zero = "00000101";
	time = ParseDate(zero, "%Y%m%d");
}

FrameCenter::FrameCenter(datetime& t, float lat, float lon, float u, float v)
{
	time = t;
	latitude = lat;
	longitude = lon;
	Um = u;
	Vm = v;
}

FrameCenter::FrameCenter(float lat, float lon, float u, float v)
{
	latitude = lat;
	longitude = lon;
	Um = u;
	Vm = v;
}

FrameCenter::~FrameCenter()
{
}

float FrameCenter::getLat() const
{
	return latitude;
}

void FrameCenter::setLat(const float& lat)
{
	latitude = lat;
}


float FrameCenter::getLon() const
{
	return longitude;
}

void FrameCenter::setLon(const float& lon)
{
	longitude = lon;
}

datetime FrameCenter::getTime() const
{
	return time;
}

void FrameCenter::setTime(const datetime& obTime)
{
	time = datetime(obTime);
}

float  FrameCenter::getUmean() const
{
	return Um;
}

void  FrameCenter::setUmean(const float& u)
{
	Um = u;
}

float  FrameCenter::getVmean() const
{
	return Vm;
}

void  FrameCenter::setVmean(const float& v)
{
	Vm = v;
}
