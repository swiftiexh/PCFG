#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main -fopenmp
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main -O1 -fopenmp
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -fopenmp

// 线程安全队列
queue<string> guess_queue;
mutex queue_mutex;
condition_variable queue_cv;
atomic<bool> finished(false);

// 哈希线程，批量取出猜测做哈希
void hash_worker(const unordered_set<string>& test_set, atomic<int>& cracked, atomic<bool>& finished, double& time_hash) {
    while (!finished || !guess_queue.empty()) {
        vector<string> batch;
        {
            unique_lock<mutex> lock(queue_mutex);
            queue_cv.wait(lock, []{ return !guess_queue.empty() || finished; });
            while (!guess_queue.empty() && batch.size() < 8) {
                batch.push_back(guess_queue.front());
                guess_queue.pop();
            }
        }
        if (!batch.empty()) {
            array<string,8> input = {"","","","","","","",""};
            for (size_t i=0; i<batch.size(); ++i) {
                input[i] = batch[i];
                if (test_set.find(batch[i]) != test_set.end()) cracked++;
            }
            bit32 state[8][4];
            auto start_hash = system_clock::now();
            MD5Hash_SIMD(input, state);
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        }
    }
}

int main()
{
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    // 加载测试数据
    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw) {
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000) break;
    }
    atomic<int> cracked(0);

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;

    // 启动哈希线程
    thread hash_thread(hash_worker, cref(test_set), ref(cracked), ref(finished), ref(time_hash));

    // PCFG生成阶段，边生成边推送到队列
    int generate_n = 10000000;
    while (!q.priority.empty()) {
        q.PopNext();
        for (const auto& guess : q.guesses) {
            unique_lock<mutex> lock(queue_mutex);
            guess_queue.push(guess);
        }
        queue_cv.notify_one();
        curr_num += q.guesses.size();
        history += q.guesses.size();
        q.guesses.clear();

        if (history >= generate_n) {
            break;
        }
    }

    // 生成结束，通知哈希线程退出
    finished = true;
    queue_cv.notify_all();
    hash_thread.join();

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
    cout << "Hash time:" << time_hash << "seconds"<<endl;
    cout << "Train time:" << time_train <<"seconds"<<endl;
    cout << "Cracked:" << cracked << endl;
}
