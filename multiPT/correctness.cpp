#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 批量序列化PT
void serializePTs(const vector<PT>& pts, vector<int>& data) {
    data.clear();
    data.push_back(pts.size());
    for (const auto& pt : pts) {
        vector<int> pt_data;
        serializePT(pt, pt_data);
        data.push_back(pt_data.size());
        data.insert(data.end(), pt_data.begin(), pt_data.end());
    }
}

// 批量反序列化PT
void deserializePTs(vector<PT>& pts, const vector<int>& data) {
    int pos = 0;
    int n = data[pos++];
    pts.resize(n);
    for (int i = 0; i < n; ++i) {
        int sz = data[pos++];
        vector<int> pt_data(data.begin() + pos, data.begin() + pos + sz);
        deserializePT(pts[i], pt_data);
        pos += sz;
    }
}

// 单个PT序列化
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

void broadcastModel(PriorityQueue& q, int rank) {
    if (rank != 0) {
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
    }
}

unordered_set<string> found;

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

    int cracked = 0;
    q.init();

    if (rank == 0) {
        cout << "here" << endl;
    }

    int total_guesses = 0;
    auto start = system_clock::now();

    const int batch_size = 8; // 可根据进程数和内存调整

    while (true) {
        bool should_continue = false;
        int actual_batch = 0;
        if (rank == 0) {
            should_continue = !q.priority.empty();
            actual_batch = min(batch_size, (int)q.priority.size());
        }
        MPI_Bcast(&should_continue, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (!should_continue) break;
        MPI_Bcast(&actual_batch, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // rank 0 批量取出PT并序列化
        vector<PT> batch_pts;
        vector<int> pt_data;
        if (rank == 0) {
            for (int i = 0; i < actual_batch; ++i) {
                batch_pts.push_back(q.priority[i]);
            }
            serializePTs(batch_pts, pt_data);
        }
        int data_size = 0;
        if (rank == 0) data_size = pt_data.size();
        MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) pt_data.resize(data_size);
        MPI_Bcast(pt_data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            deserializePTs(batch_pts, pt_data);
        }

        // 均分PT，每个进程处理一部分
        int per_proc = (actual_batch + size - 1) / size;
        int start_idx = rank * per_proc;
        int end_idx = min(start_idx + per_proc, actual_batch);

        vector<string> local_guesses;
        for (int i = start_idx; i < end_idx; ++i) {
            q.Generate(batch_pts[i]);
            local_guesses.insert(local_guesses.end(), q.guesses.begin(), q.guesses.end());
            q.guesses.clear();
        }

        // 收集所有进程的guesses到rank 0
        int local_count = local_guesses.size();
        vector<int> recv_counts(size);
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> displs(size, 0);
        int total_count = 0;
        if (rank == 0) {
            for (int i = 1; i < size; ++i)
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            total_count = displs[size - 1] + recv_counts[size - 1];
        }

        // 拼接字符串为大buffer
        string all_local;
        vector<int> str_sizes;
        for (auto &s : local_guesses) {
            str_sizes.push_back(s.size());
            all_local += s;
        }

        // 收集字符串长度
        vector<int> all_sizes;
        if (rank == 0) all_sizes.resize(total_count);
        MPI_Gatherv(str_sizes.data(), local_count, MPI_INT,
                    all_sizes.data(), recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

        // 收集字符串内容
        int local_chars = all_local.size();
        vector<int> recv_chars(size);
        MPI_Gather(&local_chars, 1, MPI_INT, recv_chars.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> char_displs(size, 0);
        int total_chars = 0;
        if (rank == 0) {
            for (int i = 1; i < size; ++i)
                char_displs[i] = char_displs[i - 1] + recv_chars[i - 1];
            total_chars = char_displs[size - 1] + recv_chars[size - 1];
        }
        string all_guesses;
        if (rank == 0) all_guesses.resize(total_chars);
        MPI_Gatherv(all_local.data(), local_chars, MPI_CHAR,
                    &all_guesses[0], recv_chars.data(), char_displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

        vector<string> batch_guesses;
        if (rank == 0) {
            int pos = 0;
            for (int i = 0; i < total_count; ++i) {
                batch_guesses.push_back(all_guesses.substr(pos, all_sizes[i]));
                pos += all_sizes[i];
            }
        }

        // 只有 rank 0 统计 cracked 和更新队列
        if (rank == 0) {
            int batch_cracked = 0;
            bit32 state[4];
            auto start_hash = system_clock::now();
            for (const string& pw : batch_guesses) {
                if (test_set.find(pw) != test_set.end() && found.find(pw) == found.end()) {
                batch_cracked++;
                found.insert(pw);
        }
                MD5Hash(pw, state);
            }
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            cracked += batch_cracked;
            total_guesses += batch_guesses.size();

            cout << "Guesses generated: " << total_guesses << ", batch_cracked: " << batch_cracked << ", total_cracked: " << cracked << endl;

            // 终止条件
            if (total_guesses >= 10000000) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                cout << "Cracked:" << cracked << endl;
                break;
            }

            // 更新优先队列
            for (int i = 0; i < actual_batch; ++i) {
                vector<PT> new_pts = q.priority[i].NewPTs();
                for (PT pt : new_pts) {
                    q.CalProb(pt);
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
            }
            q.priority.erase(q.priority.begin(), q.priority.begin() + actual_batch);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

