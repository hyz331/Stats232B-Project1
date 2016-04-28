#ifndef RGM_TIMER_HPP_
#define RGM_TIMER_HPP_

#include <string>
#include <ctime>

#include "UtilLog.hpp"

#if (defined(WIN32)  || defined(_WIN32) || defined(WIN64) || defined(_WIN64))
#define TIME( arg ) (((double) clock()) / CLOCKS_PER_SEC)
#else
#define TIME( arg ) (time( arg ))
#endif


namespace RGM
{

/// Timer
class Timers
{
public:
    struct Task {
        Task(std::string name);
        void Start();
        void Stop();
        void Reset();
        double ElapsedSeconds();

        std::string _name;
        Task* _next;
        double _offset;
        double _cumulative_time;
    };

    Timers();
    ~Timers();
    Task* operator()(std::string name);
    void showUsage();
    void clear();
private:
    Task* _head;
    Task* _tail;

    DEFINE_RGM_LOGGER;
};

} //namespace RGM

#endif // RGM_TIMER_HPP_
