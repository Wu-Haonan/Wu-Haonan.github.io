---
layout: article
title: Out-of-Order (OoO) Superscalar
tags: Computer_Architecture
aside:
  toc: true
---

This blog talks about Out-of-Order (OoO) Superscalar pipeline. We will give what hardware do in each stage.

<!--more-->

# Fetch (in program order)

Fetch multiple sequential instructions in parallel. 

# Decode (in program order)

In parallel, decode all of the instr’s just fetched. 

# Rename & Dispatch (in program order)

## Rename 

Rename the architected registers (ArchitectedRegFile, ARF) with physical
registers (PhysicalRegFile, PRF). 

### Map Table

For each architected register, we record the physical register that curretly represent this ARF. (In this class, we assume the initial map table have already assign each ARF si with a PRF pi.)

### Free List

Keep record each PRF state (used or not)

### Re-order buffer (ROB)

Holds all instructions (in order) until Commit time. In the mean time, it can contain the mapping table and overwritten list.

### How to do in "Rename" stage

* Use mapping table, replace the source archi register to physical register. 

* For dst register, pick up a available physical register to replace it.

* Record the over-written Physical Reg

* Update the free list 

Example:

lw t0, 0 (s1)

addu t0, t0, s2

mapping table 

| s1 | p1 |
| s2 | p2 |
| t0 | p3 |

Let's rename above instrs one by one.

Inst 1: lw t0, 0 (s1) --> lw p4, 0(p1) [p3]

mapping table 

| s1 | p1 |
| s2 | p2 |
| t0 | p4 |

Update p4 whcih is not available.

Inst 2: addu t0, t0, s2 --> addu p5, p4, p2 [p4]

mapping table 

| s1 | p1 |
| s2 | p2 |
| t0 | p5 |

Update p5 whcih is not available.

Note: reg in [] is over-written reg.

### When can we make Reg free again

* Over-written PRF can be freed at commit stage. (because rename and commit are in-order, so after commit, we no longer use this over-written reg.)

* Dst reg can not be frees until it's over-written. 

## Dispatch

* Make renamed instr’s eligible for execution by dispatching them to the IQ (Instruction
Queue) and the ROB (ReOrder Buffer). 

* Loads and stores are dispatched as two (micro)instr’s – one to the IQ to compute the addr and one to LSQ (LoadStoreQueue) for the
memory operation.

### Issue Queue (IQ)

Holding un-executed instrs.

An entry of IQ like

|Instr|Src1|R|Src2|R|Dst|Age|

Here "R" mean the status of source inputs (src1 Reg and scr2 Reg).

If all the srcs are ready, we can sent it to issue in next step.

Following, we will show how update IQ during dispatch stage.

Here we introduce a new list called <b>"Ready Table"</b> to record whether a PRF is ready, which provide the information for IQ.

1. For each instr, write down it as a IQ entry.

2. Get "ready status" of src PRF from Ready table.

3. Update the dst PRF as not ready in Ready Table.

### Re-order buffer (ROB)

Holds all instructions (in order) until Commit time. Recived dispatch the instrs,

|PC|Instr| PReg | AReg | Over-written | Complete| Store/Brach |
XXX0| lw |p4|t0|p3|y| |
XXX1| addu |p5|t0|p4|y| |

### Load-Store Queue (LSQ)

We store lw/sw instr in LSQ as follow

|Instr | Src | R | Address Reg | R | Dst |
|lw| | y |p1| n | p4 |
|sw| p5 | n | p1 | n||

# Issue (Out of Order)

When an instr in the IQ has all of its source data and the FU (Functional Unit) it needs
is free, it is issued for execution.

<b>update IQ</b> as following steps, select and wakeup.

## Select

Select N oldest, ready instr’s to send for execution. (And checking structual hazard)

Remove issue instr from IQ

## Wakeup

update PRF ready status

* lw / sw need to wait

* R type (has been selected) can update the dst PRF (whicn will ready next stage). So, if some instr take these dst PRFs as scr, its status is ready.

 Note: <b>Select</b> and <b>Wakeup</b> done in one cycle.

# Execution (Out of Order)

Just excute the instrs issued. 

update LSQ for src and address ready status.

# Writeback (Out of Order)

Writeback (Out Of Order - OoO): When the dst value has been computed it is written back to the PRF, the IQ, ROB and LSQ are updated – the instr <b>completes</b> execution. 

update ROB for completed Instrs.

Note: Stores DO NOT write to cache at this stage, and the ARF is NOT updated

# Commit (in program order)

Commit the instr in order, that means the oldest completed instr in ROB can be commit. (No matter how wide commit is, cannot skip over uncompleted instructions)

In Commit stage, we update ARF and store value to memory.

1. Release the first instr in <b>ROB</b> , if it <b>completed</b>.

2. Copy physical dst reg to archi dst reg.

3. Free list update over-written Reg as free

### Load-Store Queue (LSQ)

4. Stores are committed to the DM from the LSQ in program order at commit time (when they are at the head of the ROB);

5. Oldest load is “issued” for execution out of the LSQ to the DM