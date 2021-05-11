#pragma once

struct PointF
{
public:
	float X;
	float Y;
	bool IsEmpty;

	PointF()
	{
		X = -999999.9f;
		Y = -999999.9f;
		IsEmpty = true;
	}

	PointF(float x, float y)
	{
		X = x;
		Y = y;
		IsEmpty = false;
	}
};