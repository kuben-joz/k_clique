#include <iostream>

using namespace std;

int main(void)
{
    int a = 0;
    if (a++ == 0)
    {
        cout << "yes" << endl;
    }
    cout << a << endl;
}