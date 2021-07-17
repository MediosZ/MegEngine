function* generator(){
    yield 12;
    yield 123;
    yield 123;
}


const gen = generator(); // "Generator { }"

console.log(gen.next()); // 1
console.log(gen.next()); // 2
console.log(gen.next()); // 3
console.log(gen.next()); // 3
