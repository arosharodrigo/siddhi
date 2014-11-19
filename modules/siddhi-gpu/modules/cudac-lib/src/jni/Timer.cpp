#include "Timer.h"

Timer::Timer()
{
	Reset();
}

Timer::~Timer()
{
}

void Timer::Start()
{
  gettimeofday(&tValStart, NULL);
}

double Timer::Elapsed()
{
  gettimeofday(&tValEnd, NULL);
  double duration = (tValEnd.tv_sec-tValStart.tv_sec)*1000000 + tValEnd.tv_usec - tValStart.tv_usec;
  return duration/1000;
}

void Timer::Reset()
{
	tValStart = (struct timeval){0};
	tValEnd = (struct timeval){0};
}

