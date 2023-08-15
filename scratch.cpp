#include <bits/stdc++.h>

using namespace std;


template<int k>
int count() {
    std::tuple<int, int, int> tup(1,2,3);
    return std::get<k>(tup);
}

int main() {
    vector<int> a(4);
    cout << count<1>()<< endl;
    return 0;
}