#ifndef TIMER_H
#define TIMER_H

#include <stdlib.h>
#include <sys/time.h>

class Timer {

public:
  /** Constructor */
  Timer();

  /** Destructor */
  ~Timer();
  
  /** Starts the timer */
  void Start();

  /** 
   * Stops the timer.
   * Returns the duration of the timer since the last start in milliseconds.
   */
  double Elapsed();

  void Reset();

private:
  struct timeval tValStart;
  struct timeval tValEnd;
}; 

#endif
