#include <iostream>
#include <vector>
#include <string>

struct Person {
    std::string name;
    int age;
};

void greet(const Person& p) {
    std::cout << "Hello, " << p.name << "! You are " << p.age << " years old.\n";
}

int main() {
    std::vector<Person> people = {
        {"Alice", 30},
        {"Bob", 25},
        {"Josh", 28}
    };

    for (const auto& p : people) {
        greet(p);
    }

    return 0;
}