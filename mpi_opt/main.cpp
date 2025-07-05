#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;


// 序列化PT结构用于MPI广播
void serializePT(const PT& pt, vector<int>& data) {
    data.clear();
    data.push_back(pt.content.size());
    data.push_back(pt.pivot);
    data.push_back(pt.curr_indices.size());
    data.push_back(pt.max_indices.size());
    for (const auto& seg : pt.content) {
        data.push_back(seg.type);
        data.push_back(seg.length);
    }
    for (int idx : pt.curr_indices) {
        data.push_back(idx);
    }
    for (int idx : pt.max_indices) {
        data.push_back(idx);
    }
    data.push_back((int)(pt.preterm_prob * 1000000));
    data.push_back((int)(pt.prob * 1000000));
}

void deserializePT(PT& pt, const vector<int>& data) {
    int pos = 0;
    int content_size = data[pos++];
    pt.pivot = data[pos++];
    int curr_indices_size = data[pos++];
    int max_indices_size = data[pos++];
    pt.content.clear();
    for (int i = 0; i < content_size; i++) {
        int type = data[pos++];
        int length = data[pos++];
        pt.content.emplace_back(type, length);
    }
    pt.curr_indices.clear();
    for (int i = 0; i < curr_indices_size; i++) {
        pt.curr_indices.push_back(data[pos++]);
    }
    pt.max_indices.clear();
    for (int i = 0; i < max_indices_size; i++) {
        pt.max_indices.push_back(data[pos++]);
    }
    pt.preterm_prob = data[pos++] / 1000000.0f;
    pt.prob = data[pos++] / 1000000.0f;
}

// 简化的模型广播（这里假设所有进程都执行相同的训练）
void broadcastModel(PriorityQueue& q, int rank) {
    if (rank != 0) {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
}

int main()
{
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    // 训练模型
    auto start_train = system_clock::now();
    if (rank == 0) {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
    broadcastModel(q, rank);
    if (rank == 0) {
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    }

    unordered_set<std::string> test_set;
    if (rank == 0) {
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while(test_data >> pw && test_count < 1000000) {
            test_count++;
            test_set.insert(pw);
        }
    }


    q.init();

    if (rank == 0) {
        cout << "here" << endl;
    }

    int total_guesses = 0;
    auto start = system_clock::now();

    while (true) {
        bool should_continue = false;
        if (rank == 0) {
            should_continue = !q.priority.empty();
        }
        MPI_Bcast(&should_continue, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (!should_continue) {
            break;
        }

        // rank 0 弹出下一个 PT 并广播给所有进程
        PT current_pt;
        vector<int> pt_data;
        int data_size = 0;
        if (rank == 0) {
            current_pt = q.priority.front();
            serializePT(current_pt, pt_data);
            data_size = pt_data.size();
        }
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            pt_data.resize(data_size);
        }
        MPI_Bcast(pt_data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            deserializePT(current_pt, pt_data);
        }

        // 记录生成前的猜测数量
        int prev_total = q.total_guesses;

        // 所有进程参与生成
        q.Generate(current_pt);

        // 只有 rank 0 更新队列
        if (rank == 0) {
            bit32 state[4];
            auto start_hash = system_clock::now();
            for (const string& pw : q.guesses) {
                MD5Hash(pw, state);
            }
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            total_guesses += q.guesses.size();

            cout << "Guesses generated: " << total_guesses;

            // 清空本批 guesses，避免内存暴涨
            q.guesses.clear();

            // 终止条件
            if (total_guesses >= 10000000) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }

            // 更新优先队列
            vector<PT> new_pts = q.priority.front().NewPTs();
            for (PT pt : new_pts) {
                q.CalProb(pt);
                // 插入逻辑保持不变
                bool inserted = false;
                for (auto iter = q.priority.begin(); iter != q.priority.end(); iter++) {
                    if (iter != q.priority.end() - 1 && iter != q.priority.begin()) {
                        if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                            q.priority.emplace(iter + 1, pt);
                            inserted = true;
                            break;
                        }
                    }
                    if (iter == q.priority.end() - 1) {
                        q.priority.emplace_back(pt);
                        inserted = true;
                        break;
                    }
                    if (iter == q.priority.begin() && iter->prob < pt.prob) {
                        q.priority.emplace(iter, pt);
                        inserted = true;
                        break;
                    }
                }
            }
            q.priority.erase(q.priority.begin());
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}