const std = @import("std");
const Thread = std.Thread;

pub const Queue = struct {
    pub fn FIFO(comptime T: type) type {
        const Node = struct {
            value: T,
            next: ?*@This(),
        };

        return struct {
            const Self = @This();
            mutex: Thread.Mutex = .{},
            allocator: std.mem.Allocator,

            head: ?*Node = null,
            tail: ?*Node = null,

            pub fn init(allocator: std.mem.Allocator) Self {
                return Self{
                    .allocator = allocator,
                };
            }

            pub fn deinit(self: *Self) void {
                while (self.dequeue()) |_| {}
            }

            pub fn dequeue(self: *Self) ?T {
                self.mutex.lock();
                defer self.mutex.unlock();

                if (self.head) |node| {
                    const value = node.value;
                    self.head = node.next;
                    if (self.head == null) {
                        self.tail = null;
                    }
                    self.allocator.destroy(node);
                    return value;
                }

                return null;
            }

            pub fn enqueue(self: *Self, value: T) !void {
                const node = try self.allocator.create(Node);
                node.* = .{ .value = value, .next = null };

                self.mutex.lock();
                defer self.mutex.unlock();

                if (self.tail) |tail| {
                    tail.next = node;
                } else {
                    self.head = node;
                }
                self.tail = node;
            }
        };
    }
};

const testing = std.testing;

test "FIFO Queue - basic operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var queue = Queue.FIFO(i32).init(allocator);
    defer queue.deinit();

    // Test enqueue
    try queue.enqueue(1);
    try queue.enqueue(2);
    try queue.enqueue(3);

    // Test dequeue
    try testing.expectEqual(@as(?i32, 1), queue.dequeue());
    try testing.expectEqual(@as(?i32, 2), queue.dequeue());
    try testing.expectEqual(@as(?i32, 3), queue.dequeue());

    // Test empty queue
    try testing.expectEqual(@as(?i32, null), queue.dequeue());
}

test "FIFO Queue - interleaved enqueue and dequeue" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var queue = Queue.FIFO(i32).init(allocator);
    defer queue.deinit();

    try queue.enqueue(1);
    try queue.enqueue(2);
    try testing.expectEqual(@as(?i32, 1), queue.dequeue());
    try queue.enqueue(3);
    try testing.expectEqual(@as(?i32, 2), queue.dequeue());
    try testing.expectEqual(@as(?i32, 3), queue.dequeue());
    try testing.expectEqual(@as(?i32, null), queue.dequeue());
}

const ThreadContext = struct {
    queue: *Queue.FIFO(i32),
    start_value: i32,
    end_value: i32,
    allocator: std.mem.Allocator,
};

fn enqueueWorker(context: *ThreadContext) !void {
    var i = context.start_value;
    while (i <= context.end_value) : (i += 1) {
        try context.queue.enqueue(i);
    }
}

test "FIFO Queue - concurrent access" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var queue = Queue.FIFO(i32).init(allocator);
    defer queue.deinit();

    var context1 = ThreadContext{ .queue = &queue, .start_value = 1, .end_value = 1000, .allocator = allocator };
    var context2 = ThreadContext{ .queue = &queue, .start_value = 1001, .end_value = 2000, .allocator = allocator };

    const thread1 = try std.Thread.spawn(.{}, enqueueWorker, .{&context1});
    const thread2 = try std.Thread.spawn(.{}, enqueueWorker, .{&context2});

    thread1.join();
    thread2.join();

    var sum: i32 = 0;
    var count: usize = 0;
    while (queue.dequeue()) |value| {
        sum += value;
        count += 1;
    }

    try testing.expectEqual(@as(usize, 2000), count);
    try testing.expectEqual(@as(i32, 2001000), sum); // Sum of numbers from 1 to 2000
}
