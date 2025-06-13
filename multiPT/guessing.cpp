#include "PCFG.h"
#include <mpi.h>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
    return res;
}

// 单个PT的Generate不变
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int total = pt.max_indices[0];
        int chunk = (total + size - 1) / size;
        int start = rank * chunk;
        int end = std::min(start + chunk, total);

        vector<string> local_guesses;
        for (int i = start; i < end; i++)
        {
            string guess = a->ordered_values[i];
            local_guesses.emplace_back(guess);
        }

        int local_count = local_guesses.size();
        vector<int> recv_counts(size);
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        string all_local;
        vector<int> str_sizes;
        for (auto &s : local_guesses)
        {
            str_sizes.push_back(s.size());
            all_local += s;
        }
        vector<int> displs(size, 0);
        int total_count = 0;
        if (rank == 0)
        {
            for (int i = 1; i < size; ++i)
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            total_count = displs[size - 1] + recv_counts[size - 1];
            guesses.resize(total_count);
        }
        vector<int> all_sizes;
        if (rank == 0) all_sizes.resize(total_count);
        MPI_Gatherv(str_sizes.data(), local_count, MPI_INT,
                    all_sizes.data(), recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        int local_chars = all_local.size();
        vector<int> recv_chars(size);
        MPI_Gather(&local_chars, 1, MPI_INT, recv_chars.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> char_displs(size, 0);
        int total_chars = 0;
        if (rank == 0)
        {
            for (int i = 1; i < size; ++i)
                char_displs[i] = char_displs[i - 1] + recv_chars[i - 1];
            total_chars = char_displs[size - 1] + recv_chars[size - 1];
        }
        string all_guesses;
        if (rank == 0) all_guesses.resize(total_chars);
        MPI_Gatherv(all_local.data(), local_chars, MPI_CHAR,
                    &all_guesses[0], recv_chars.data(), char_displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            int pos = 0;
            for (int i = 0; i < total_count; ++i)
            {
                guesses.push_back(all_guesses.substr(pos, all_sizes[i]));
                pos += all_sizes[i];
            }
            total_guesses += total_count;
        }
        MPI_Bcast(&total_guesses, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];

        int total = pt.max_indices[pt.content.size() - 1];
        int chunk = (total + size - 1) / size;
        int start = rank * chunk;
        int end = std::min(start + chunk, total);

        vector<string> local_guesses;
        for (int i = start; i < end; i++)
        {
            string temp = guess + a->ordered_values[i];
            local_guesses.emplace_back(temp);
        }

        int local_count = local_guesses.size();
        vector<int> recv_counts(size);
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> displs(size, 0);
        int total_count = 0;
        if (rank == 0)
        {
            for (int i = 1; i < size; ++i)
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            total_count = displs[size - 1] + recv_counts[size - 1];
            guesses.resize(total_count);
        }
        string all_local;
        vector<int> str_sizes;
        for (auto &s : local_guesses)
        {
            str_sizes.push_back(s.size());
            all_local += s;
        }
        vector<int> all_sizes;
        if (rank == 0) all_sizes.resize(total_count);
        MPI_Gatherv(str_sizes.data(), local_count, MPI_INT,
                    all_sizes.data(), recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        int local_chars = all_local.size();
        vector<int> recv_chars(size);
        MPI_Gather(&local_chars, 1, MPI_INT, recv_chars.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        vector<int> char_displs(size, 0);
        int total_chars = 0;
        if (rank == 0)
        {
            for (int i = 1; i < size; ++i)
                char_displs[i] = char_displs[i - 1] + recv_chars[i - 1];
            total_chars = char_displs[size - 1] + recv_chars[size - 1];
        }
        string all_guesses;
        if (rank == 0) all_guesses.resize(total_chars);
        MPI_Gatherv(all_local.data(), local_chars, MPI_CHAR,
                    &all_guesses[0], recv_chars.data(), char_displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            int pos = 0;
            for (int i = 0; i < total_count; ++i)
            {
                guesses.push_back(all_guesses.substr(pos, all_sizes[i]));
                pos += all_sizes[i];
            }
            total_guesses += total_count;
        }
        MPI_Bcast(&total_guesses, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
}