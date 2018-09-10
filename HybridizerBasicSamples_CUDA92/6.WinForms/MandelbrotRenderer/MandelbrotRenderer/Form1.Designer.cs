using System.Windows.Forms;

namespace MandelbrotRenderer
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.MaxiterInput = new System.Windows.Forms.NumericUpDown();
            this.RenderButton = new System.Windows.Forms.Button();
            this.Flavor = new System.Windows.Forms.GroupBox();
            this.AVX512 = new System.Windows.Forms.RadioButton();
            this.AVX2 = new System.Windows.Forms.RadioButton();
            this.AVX = new System.Windows.Forms.RadioButton();
            this.CUDA = new System.Windows.Forms.RadioButton();
            this.CSharp = new System.Windows.Forms.RadioButton();
            this.Rendering = new System.Windows.Forms.PictureBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.MaxiterInput)).BeginInit();
            this.Flavor.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Rendering)).BeginInit();
            this.SuspendLayout();
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.FixedPanel = System.Windows.Forms.FixedPanel.Panel1;
            this.splitContainer1.IsSplitterFixed = true;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.label7);
            this.splitContainer1.Panel1.Controls.Add(this.label6);
            this.splitContainer1.Panel1.Controls.Add(this.label5);
            this.splitContainer1.Panel1.Controls.Add(this.label4);
            this.splitContainer1.Panel1.Controls.Add(this.label3);
            this.splitContainer1.Panel1.Controls.Add(this.label2);
            this.splitContainer1.Panel1.Controls.Add(this.label1);
            this.splitContainer1.Panel1.Controls.Add(this.MaxiterInput);
            this.splitContainer1.Panel1.Controls.Add(this.RenderButton);
            this.splitContainer1.Panel1.Controls.Add(this.Flavor);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.Rendering);
            this.splitContainer1.Size = new System.Drawing.Size(925, 786);
            this.splitContainer1.SplitterDistance = 180;
            this.splitContainer1.TabIndex = 0;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F);
            this.label5.Location = new System.Drawing.Point(9, 279);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(54, 13);
            this.label5.TabIndex = 9;
            this.label5.Text = "gpu name";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F);
            this.label4.Location = new System.Drawing.Point(9, 255);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(54, 13);
            this.label4.TabIndex = 8;
            this.label4.Text = "cpu name";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 230);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(51, 13);
            this.label3.TabIndex = 7;
            this.label3.Text = "Size : 4.0";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(9, 206);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(82, 13);
            this.label2.TabIndex = 6;
            this.label2.Text = "Rendering Time";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(9, 158);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(41, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "Maxiter";
            // 
            // MaxiterInput
            // 
            this.MaxiterInput.Location = new System.Drawing.Point(104, 156);
            this.MaxiterInput.Maximum = new decimal(new int[] {
            1000000,
            0,
            0,
            0});
            this.MaxiterInput.Name = "MaxiterInput";
            this.MaxiterInput.Size = new System.Drawing.Size(76, 20);
            this.MaxiterInput.TabIndex = 4;
            this.MaxiterInput.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.MaxiterInput.ValueChanged += new System.EventHandler(this.MaxiterInput_ValueChanged);
            // 
            // RenderButton
            // 
            this.RenderButton.Location = new System.Drawing.Point(12, 182);
            this.RenderButton.Name = "RenderButton";
            this.RenderButton.Size = new System.Drawing.Size(165, 21);
            this.RenderButton.TabIndex = 2;
            this.RenderButton.Text = "Render";
            this.RenderButton.UseVisualStyleBackColor = true;
            this.RenderButton.Click += new System.EventHandler(this.RenderButton_Click);
            // 
            // Flavor
            // 
            this.Flavor.Controls.Add(this.AVX512);
            this.Flavor.Controls.Add(this.AVX2);
            this.Flavor.Controls.Add(this.AVX);
            this.Flavor.Controls.Add(this.CUDA);
            this.Flavor.Controls.Add(this.CSharp);
            this.Flavor.Location = new System.Drawing.Point(12, 12);
            this.Flavor.Name = "Flavor";
            this.Flavor.Size = new System.Drawing.Size(165, 143);
            this.Flavor.TabIndex = 1;
            this.Flavor.TabStop = false;
            this.Flavor.Text = "Flavor";
            // 
            // AVX512
            // 
            this.AVX512.AutoSize = true;
            this.AVX512.Location = new System.Drawing.Point(9, 111);
            this.AVX512.Name = "AVX512";
            this.AVX512.Size = new System.Drawing.Size(64, 17);
            this.AVX512.TabIndex = 5;
            this.AVX512.Text = "AVX512";
            this.AVX512.UseVisualStyleBackColor = true;
            // 
            // AVX2
            // 
            this.AVX2.AutoSize = true;
            this.AVX2.Location = new System.Drawing.Point(9, 88);
            this.AVX2.Name = "AVX2";
            this.AVX2.Size = new System.Drawing.Size(52, 17);
            this.AVX2.TabIndex = 4;
            this.AVX2.Text = "AVX2";
            this.AVX2.UseVisualStyleBackColor = true;
            this.AVX2.CheckedChanged += new System.EventHandler(this.FlavorCheckedChanged);
            // 
            // AVX
            // 
            this.AVX.AutoSize = true;
            this.AVX.Location = new System.Drawing.Point(9, 65);
            this.AVX.Name = "AVX";
            this.AVX.Size = new System.Drawing.Size(46, 17);
            this.AVX.TabIndex = 2;
            this.AVX.Text = "AVX";
            this.AVX.UseVisualStyleBackColor = true;
            this.AVX.CheckedChanged += new System.EventHandler(this.FlavorCheckedChanged);
            // 
            // CUDA
            // 
            this.CUDA.AutoSize = true;
            this.CUDA.Location = new System.Drawing.Point(9, 42);
            this.CUDA.Name = "CUDA";
            this.CUDA.Size = new System.Drawing.Size(55, 17);
            this.CUDA.TabIndex = 1;
            this.CUDA.Text = "CUDA";
            this.CUDA.UseVisualStyleBackColor = true;
            this.CUDA.CheckedChanged += new System.EventHandler(this.FlavorCheckedChanged);
            // 
            // CSharp
            // 
            this.CSharp.AutoSize = true;
            this.CSharp.Checked = true;
            this.CSharp.Location = new System.Drawing.Point(9, 19);
            this.CSharp.Name = "CSharp";
            this.CSharp.Size = new System.Drawing.Size(39, 17);
            this.CSharp.TabIndex = 0;
            this.CSharp.TabStop = true;
            this.CSharp.Text = "C#";
            this.CSharp.UseVisualStyleBackColor = true;
            this.CSharp.CheckedChanged += new System.EventHandler(this.FlavorCheckedChanged);
            // 
            // Rendering
            // 
            this.Rendering.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.Rendering.Dock = System.Windows.Forms.DockStyle.Fill;
            this.Rendering.Location = new System.Drawing.Point(0, 0);
            this.Rendering.Name = "Rendering";
            this.Rendering.Size = new System.Drawing.Size(741, 786);
            this.Rendering.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.Rendering.TabIndex = 0;
            this.Rendering.TabStop = false;
            this.Rendering.Click += new System.EventHandler(this.Rendering_Click);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(9, 317);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(99, 13);
            this.label6.TabIndex = 10;
            this.label6.Text = "Zoom in   : left click";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(9, 330);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(106, 13);
            this.label7.TabIndex = 11;
            this.label7.Text = "Zoom out : right click";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(925, 786);
            this.Controls.Add(this.splitContainer1);
            this.Name = "Form1";
            this.Text = "Mandelbrot Renderer";
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel1.PerformLayout();
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.MaxiterInput)).EndInit();
            this.Flavor.ResumeLayout(false);
            this.Flavor.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Rendering)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.PictureBox Rendering;
        private System.Windows.Forms.RadioButton AVX2;
        private System.Windows.Forms.RadioButton AVX;
        private System.Windows.Forms.RadioButton CUDA;
        private System.Windows.Forms.RadioButton CSharp;
        private System.Windows.Forms.GroupBox Flavor;
        private Button RenderButton;
        private NumericUpDown MaxiterInput;
        private Label label1;
        private Label label2;
        private Label label3;
        private Label label5;
        private Label label4;
        private RadioButton AVX512;
        private Label label7;
        private Label label6;
    }
}

