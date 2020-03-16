#ifndef _STOPWATCH_HPP_
#define _STOPWATCH_HPP_


#ifdef WIN32
	#include <windows.h>
#else
	#include <unistd.h>
	#include <time.h>
	#include <sys/times.h>
	#include <sys/time.h>
#endif


/**
 * \brief Implementation of high precision wall-time stopwatch based on system timers.
 */
class Stopwatch
{
private:
	typedef unsigned long long ticks_t;

	ticks_t mStartTime;
	double mInterval;
	bool mTiming;

	/**
	 * \brief Get current system timer status in ticks.
	 */
	ticks_t now()
	{
#ifdef WIN32
		LARGE_INTEGER ticks;
		::QueryPerformanceCounter(&ticks);
		return static_cast<ticks_t>(ticks.QuadPart);
#else
		struct timespec ts;
		::clock_gettime(CLOCK_REALTIME, &ts);
		return static_cast<ticks_t>(ts.tv_sec) * 1000000000UL + static_cast<ticks_t>(ts.tv_nsec);
#endif
	}


	/**
	 * Measure current time and update mInterval.
	 */
	void measureTime()
	{
#ifdef WIN32
		LARGE_INTEGER ticks;
		::QueryPerformanceFrequency(&ticks);
		mInterval += static_cast<double>(now() - mStartTime) / static_cast<double>(ticks.QuadPart);
#else
		mInterval += static_cast<double>((now() - mStartTime)*1E-9);
#endif
	}


public:
	/**
	 * \brief Create new stopwatch. The stopwatch are not running when created.
	 */
	Stopwatch() : mTiming(false), mInterval(0.0) { }

	/**
	 * \brief Create new stopwatch (and optionaly start it).
	 * \param start If start is true, the stapwatch are started immediately.
	 */
	Stopwatch(bool start)
	{
		if (start) this->start();
	}


	/**
	 * \brief Start the stopwatch. If the stopwatch are already timing, they are reset.
	 */
	void start()
	{
		mTiming = true;
		mStartTime = now();
	}


	/**
	 * \brief Stop the stopwatch. Multiple invocation has no effect.
	 */
	void stop()
	{
		if (mTiming == false) return;
		mTiming = false;
		measureTime();
	}
	
	/**
	 * \brief Stop and reset the stopwatch. Multiple invocation has no effect.
	 */
	void reset()
	{
		mInterval = 0.0;
		if (mTiming == false) return;
		mTiming = false;
	}


	/**
	 * \brief Return measured time in seconds.
	 */
	double getSeconds()
	{
		if (mTiming)
			measureTime();
		return mInterval;
	}

	/**
	 * \brief Return mesured time in miliseconds.
	 */
	double getMiliseconds()
	{
		return getSeconds() * 1000.0;
	}
};

#endif
