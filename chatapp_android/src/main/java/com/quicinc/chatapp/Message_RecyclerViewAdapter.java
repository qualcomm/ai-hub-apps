// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.chatapp;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class Message_RecyclerViewAdapter extends RecyclerView.Adapter<Message_RecyclerViewAdapter.MyViewHolder> {

    Context context;
    ArrayList<ChatMessage> messages = new ArrayList<ChatMessage>(1000);

    public Message_RecyclerViewAdapter(Context context, ArrayList<ChatMessage> messages) {
        this.context = context;
        this.messages = messages;
    }

    @NonNull
    @Override
    public Message_RecyclerViewAdapter.MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(context);
        View view = inflater.inflate(R.layout.chat_row, parent, false);

        return new Message_RecyclerViewAdapter.MyViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull Message_RecyclerViewAdapter.MyViewHolder holder, int position) {
        ChatMessage msg = messages.get(position);
        if (msg.isMessageFromUser()) {
            holder.mUserMessage.setText(msg.getMessage());
            holder.mLeftChatLayout.setVisibility(View.GONE);
            holder.mRightChatLayout.setVisibility(View.VISIBLE);
        } else {
            holder.mBotMessage.setText(msg.getMessage());
            holder.mLeftChatLayout.setVisibility(View.VISIBLE);
            holder.mRightChatLayout.setVisibility(View.GONE);
        }
    }

    @Override
    public int getItemCount() {
        return messages.size();
    }

    public void addMessage(ChatMessage msg) {
        messages.add(msg);
    }

    /**
     * updateBotMessage: updates / inserts message on behalf of Bot
     *
     * @param bot_message message to update or insert
     * @return newly added message
     */
    public String updateBotMessage(String bot_message) {
        boolean lastMessageFromBot = false;
        ChatMessage lastMessage;

        if (messages.size() > 1) {
            lastMessage = messages.get(messages.size() - 1);
            if (lastMessage.mSender == MessageSender.BOT) {
                lastMessageFromBot = true;
            }
        } else {
            addMessage(new ChatMessage(bot_message, MessageSender.BOT));
        }

        if (lastMessageFromBot) {
            messages.get(messages.size() - 1).mMessage = messages.get(messages.size() - 1).mMessage + bot_message;
        } else {
            addMessage(new ChatMessage(bot_message, MessageSender.BOT));
        }
        return messages.get(messages.size() - 1).mMessage;
    }

    public static class MyViewHolder extends RecyclerView.ViewHolder {

        TextView mUserMessage;
        TextView mBotMessage;
        LinearLayout mLeftChatLayout;
        LinearLayout mRightChatLayout;

        public MyViewHolder(@NonNull View itemView) {
            super(itemView);

            mBotMessage = itemView.findViewById(R.id.bot_message);
            mUserMessage = itemView.findViewById(R.id.user_message);
            mLeftChatLayout = itemView.findViewById(R.id.left_chat_layout);
            mRightChatLayout = itemView.findViewById(R.id.right_chat_layout);
        }
    }
}
